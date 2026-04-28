"""
Multi-game parallel self-play for high GPU utilization.

Problem with sequential self-play:
  Each MCTS search produces batch=32 leaves → GPU is ~20% utilized.
  Between batches, Python does tree traversal, move gen, cloning → GPU idle.

Solution: run N games SIMULTANEOUSLY, collect leaves from ALL games
into ONE mega-batch, do ONE GPU call:
  8 concurrent games × 32 leaves each = 256 leaves per GPU call
  GPU utilization jumps from ~20% to ~80%+

Architecture:
  ParallelSelfPlayWorker manages N "game slots". Each slot has its own
  MCTS tree. Per simulation round:
    1. For each active game, traverse tree to find leaves (with virtual loss)
    2. Stack ALL leaves from ALL games into one tensor
    3. ONE model.forward() call
    4. Distribute results back, expand nodes, backpropagate
  When a game finishes its MCTS budget → pick move, apply, continue.
  When a game ends → harvest training data, start a new game in that slot.
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from game import AntichessGame, Move, ACTION_SIZE
from mcts import MCTSNode, C_PUCT, DIR_ALPHA, DIR_EPSILON


@dataclass
class GameStats:
    """Statistics from a single self-play game."""
    num_moves: int = 0
    resigned: bool = False
    winner: int = -1


@dataclass
class GameSlot:
    """One concurrent game being played."""
    game: AntichessGame = None
    trajectory: list = field(default_factory=list)  # (state, pi, player)
    move_num: int = 0
    resign_enabled: bool = True
    consecutive_low: list = field(default_factory=lambda: [0, 0])
    # MCTS state for current move
    mcts_root: MCTSNode = None
    mcts_sims_done: int = 0

    def reset(self, random_opening_moves: int = 0, resign_disabled_frac: float = 0.1):
        self.game = AntichessGame()
        self.trajectory = []
        self.move_num = 0
        self.resign_enabled = np.random.random() > resign_disabled_frac
        self.consecutive_low = [0, 0]
        self.mcts_root = None
        self.mcts_sims_done = 0


class ParallelSelfPlayWorker:
    """
    Plays N games simultaneously with shared GPU batching.

    Key insight: instead of
        for game in games:         # sequential
            mcts.search(game)      # each produces small GPU batch

    We do:
        for sim_round in range(sims_per_batch):
            leaves = []
            for game in games:     # collect from ALL games
                leaves += traverse_tree(game)
            model.forward(leaves)  # ONE big GPU call
            backprop_all(leaves)

    Args:
        model: neural network
        num_simulations: MCTS sims per move
        num_parallel: number of concurrent games (tune to GPU memory)
        leaves_per_game: leaves to collect per game per batch round
        temp_threshold: move number after which temperature drops
        random_opening_moves: random moves at start for diversity
        resign_threshold: resign below this value
        resign_consecutive: for this many consecutive moves
    """

    def __init__(self, model, num_simulations: int = 800,
                 num_parallel: int = 8, leaves_per_game: int = 8,
                 temp_threshold: int = 30,
                 random_opening_moves: int = 0,
                 resign_threshold: float = -0.95,
                 resign_consecutive: int = 5,
                 resign_disabled_frac: float = 0.1):
        self.model = model
        self.num_simulations = num_simulations
        self.num_parallel = num_parallel
        self.leaves_per_game = leaves_per_game
        self.temp_threshold = temp_threshold
        self.random_opening_moves = random_opening_moves
        self.resign_threshold = resign_threshold
        self.resign_consecutive = resign_consecutive
        self.resign_disabled_frac = resign_disabled_frac
        self.device = next(model.parameters()).device

    def play_games(self, total_games: int) -> Tuple[List[Tuple], List[GameStats]]:
        """
        Play `total_games` games using N parallel slots.

        Returns:
            all_examples: (state, pi, value) training tuples with augmentation
            all_stats: per-game statistics
        """
        all_examples = []
        all_stats = []
        games_completed = 0

        # Initialize parallel game slots
        slots = []
        for _ in range(min(self.num_parallel, total_games)):
            slot = GameSlot()
            slot.reset(self.random_opening_moves, self.resign_disabled_frac)
            slots.append(slot)

        # Handle random opening moves for initial slots
        for slot in slots:
            self._do_random_opening(slot)

        while games_completed < total_games:
            # ── Step 1: Ensure each slot has an MCTS root for its current move ──
            roots_to_init = []
            roots_slots = []
            for slot in slots:
                if slot.game is None:
                    continue
                over, _ = slot.game.is_game_over()
                if over:
                    # Harvest finished game
                    examples, stats = self._finalize_game(slot)
                    all_examples.extend(examples)
                    all_stats.append(stats)
                    games_completed += 1
                    if (games_completed) % 5 == 0 or games_completed == total_games:
                        print(f"  Games: {games_completed}/{total_games}, "
                              f"examples: {len(all_examples)}", flush=True)
                    if games_completed < total_games:
                        slot.reset(self.random_opening_moves, self.resign_disabled_frac)
                        self._do_random_opening(slot)
                    else:
                        slot.game = None
                    continue

                if slot.mcts_root is None:
                    # Need to initialize MCTS root
                    root = MCTSNode(slot.game.clone())
                    slot.mcts_root = root
                    slot.mcts_sims_done = 0
                    roots_to_init.append(root)
                    roots_slots.append(slot)

            # Batch-initialize roots
            if roots_to_init:
                self._batch_expand(roots_to_init)
                for root, slot in zip(roots_to_init, roots_slots):
                    # Add Dirichlet noise
                    if root.children:
                        noise = np.random.dirichlet([DIR_ALPHA] * len(root.children))
                        for i, child in enumerate(root.children):
                            child.prior = ((1 - DIR_EPSILON) * child.prior
                                           + DIR_EPSILON * noise[i])

            # ── Step 2: Run MCTS simulation rounds across all active games ──
            active_slots = [s for s in slots if s.game is not None
                            and s.mcts_root is not None
                            and s.mcts_sims_done < self.num_simulations]

            if not active_slots:
                continue

            # Do simulation rounds until all active slots hit their budget
            sims_per_round = self.leaves_per_game  # leaves per game per round

            while active_slots:
                all_leaves = []       # (MCTSNode, slot_index)
                all_states = []       # encoded states
                all_terminals = []    # (MCTSNode, value, slot_index)

                for slot in active_slots:
                    root = slot.mcts_root
                    for _ in range(sims_per_round):
                        # SELECT
                        node = root
                        while node.is_expanded and node.children:
                            node = max(node.children, key=lambda c: c.ucb_score())

                        node.add_virtual_loss()

                        # Terminal?
                        game_over, winner = node.game.is_game_over()
                        if game_over:
                            if winner == -1:
                                val = 0.0
                            elif winner == slot.game.turn:
                                val = 1.0
                            else:
                                val = -1.0
                            all_terminals.append((node, val))
                        elif node.is_expanded:
                            all_terminals.append((node, 0.0))
                        else:
                            all_leaves.append(node)
                            all_states.append(node.game.encode())

                # ── ONE GPU call for ALL leaves from ALL games ──
                if all_states:
                    policies, values = self._batch_predict(all_states)
                    for i, node in enumerate(all_leaves):
                        self._expand_with_policy(node, policies[i])
                        self._backpropagate(node, float(values[i]))
                        node.remove_virtual_loss()

                for node, val in all_terminals:
                    self._backpropagate(node, val)
                    node.remove_virtual_loss()

                # Update sim counts
                for slot in active_slots:
                    slot.mcts_sims_done += sims_per_round

                # Remove slots that hit their budget
                active_slots = [s for s in active_slots
                                if s.mcts_sims_done < self.num_simulations]

            # ── Step 3: Pick moves for all slots that finished MCTS ──
            for slot in slots:
                if slot.game is None or slot.mcts_root is None:
                    continue
                if slot.mcts_sims_done < self.num_simulations:
                    continue

                root = slot.mcts_root
                temperature = 1.0 if slot.move_num < self.temp_threshold else 0.1
                pi = self._extract_policy(root, temperature)

                # Record
                state = slot.game.encode()
                slot.trajectory.append((state, pi, slot.game.turn))

                # Resignation check
                resigned = False
                if (slot.resign_enabled and
                        slot.move_num >= self.random_opening_moves + 10):
                    _, value = self._single_predict(state)
                    side = slot.game.turn
                    if value < self.resign_threshold:
                        slot.consecutive_low[side] += 1
                    else:
                        slot.consecutive_low[side] = 0
                    if slot.consecutive_low[side] >= self.resign_consecutive:
                        resigned = True

                if resigned:
                    # Finalize as resignation
                    winner = 1 - slot.game.turn
                    examples, stats = self._finalize_game(slot, forced_winner=winner,
                                                          resigned=True)
                    all_examples.extend(examples)
                    all_stats.append(stats)
                    games_completed += 1
                    if games_completed < total_games:
                        slot.reset(self.random_opening_moves, self.resign_disabled_frac)
                        self._do_random_opening(slot)
                    else:
                        slot.game = None
                    continue

                # Apply move
                action = np.random.choice(ACTION_SIZE, p=pi)
                legal = slot.game.legal_moves()
                move = None
                for m in legal:
                    if m.to_action_index() == action:
                        move = m
                        break
                if move is None:
                    move = legal[np.random.randint(len(legal))]

                slot.game.apply_move(move)
                slot.move_num += 1
                slot.mcts_root = None  # reset for next move
                slot.mcts_sims_done = 0

        return all_examples, all_stats

    # ─── Helper Methods ───

    def _do_random_opening(self, slot: GameSlot):
        """Play random opening moves."""
        while slot.move_num < self.random_opening_moves:
            over, _ = slot.game.is_game_over()
            if over:
                break
            legal = slot.game.legal_moves()
            move = legal[np.random.randint(len(legal))]
            state = slot.game.encode()
            pi = np.zeros(ACTION_SIZE, dtype=np.float32)
            pi[move.to_action_index()] = 1.0
            slot.trajectory.append((state, pi, slot.game.turn))
            slot.game.apply_move(move)
            slot.move_num += 1

    def _finalize_game(self, slot: GameSlot, forced_winner=None,
                       resigned=False) -> Tuple[List[Tuple], GameStats]:
        """Harvest training data from a finished game."""
        if forced_winner is not None:
            winner = forced_winner
        else:
            _, winner = slot.game.is_game_over()

        stats = GameStats(
            num_moves=slot.move_num,
            resigned=resigned,
            winner=winner if winner is not None else -1,
        )

        examples = []
        for state, pi, player in slot.trajectory:
            if winner == -1:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            examples.append((state, pi, value))

            # Augmentation: horizontal flip
            flipped_state = np.flip(state, axis=2).copy()
            flipped_pi = _flip_policy(pi)
            examples.append((flipped_state, flipped_pi, value))

        return examples, stats

    def _batch_predict(self, states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Batch GPU inference — the hot path."""
        self.model.eval()
        with torch.no_grad():
            batch = torch.FloatTensor(np.array(states)).to(self.device)
            logits, v = self.model(batch)
            policies = F.softmax(logits, dim=1).cpu().numpy()
            values = v.squeeze(-1).cpu().numpy()
        return policies, values

    def _single_predict(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Single state prediction (for resignation check)."""
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, v = self.model(x)
            policy = F.softmax(logits, dim=1).cpu().numpy()[0]
            value = v.cpu().item()
        return policy, value

    def _batch_expand(self, nodes: List[MCTSNode]):
        """Expand multiple root nodes in one batch."""
        states = [n.game.encode() for n in nodes]
        policies, _ = self._batch_predict(states)
        for node, policy in zip(nodes, policies):
            node.is_expanded = True
            moves = node.game.legal_moves()
            if moves:
                self._create_children(node, moves, policy)

    def _expand_with_policy(self, node: MCTSNode, policy: np.ndarray):
        if node.is_expanded:
            return
        node.is_expanded = True
        moves = node.game.legal_moves()
        if moves:
            self._create_children(node, moves, policy)

    def _create_children(self, node: MCTSNode, moves: List[Move],
                         policy: np.ndarray):
        legal_priors = [(m, policy[m.to_action_index()]) for m in moves]
        total = sum(p for _, p in legal_priors)
        if total < 1e-8:
            total = 1.0
        for move, prior in legal_priors:
            child_game = node.game.clone()
            child_game.apply_move(move)
            child = MCTSNode(child_game, move=move, parent=node,
                             prior=prior / total)
            node.children.append(child)

    def _backpropagate(self, node: MCTSNode, value: float):
        current = node
        depth = 0
        while current is not None:
            current.visit_count += 1
            if depth % 2 == 0:
                current.total_value += value
            else:
                current.total_value -= value
            current = current.parent
            depth += 1

    def _extract_policy(self, root: MCTSNode, temperature: float) -> np.ndarray:
        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        for child in root.children:
            pi[child.move.to_action_index()] = child.visit_count
        if pi.sum() == 0:
            return pi
        if temperature < 1e-4:
            best = np.argmax(pi)
            pi = np.zeros_like(pi)
            pi[best] = 1.0
        else:
            pi = pi ** (1.0 / temperature)
            pi /= pi.sum()
        return pi


# ─── Policy flip (unchanged) ───

def _flip_policy(pi: np.ndarray) -> np.ndarray:
    from game import rc, idx
    new_pi = np.zeros_like(pi)
    for a in range(ACTION_SIZE):
        if a < 4096:
            fsq, tsq = a // 64, a % 64
            fr, fc = rc(fsq)
            tr, tc = rc(tsq)
            new_a = idx(fr, 7 - fc) * 64 + idx(tr, 7 - tc)
            new_pi[new_a] = pi[a]
        else:
            offset = a - 4096
            col = offset // 16
            rest = offset % 16
            new_a = 4096 + (7 - col) * 16 + rest
            new_pi[new_a] = pi[a]
    return new_pi


# ─── Legacy compatible wrapper ───

class SelfPlayWorker:
    """Backward-compatible wrapper. Used by train.py's parallel_self_play."""

    def __init__(self, model, num_simulations=800, temp_threshold=30,
                 random_opening_moves=0, resign_threshold=-0.95,
                 resign_consecutive=5, resign_disabled_frac=0.1):
        self.worker = ParallelSelfPlayWorker(
            model, num_simulations=num_simulations,
            num_parallel=8, leaves_per_game=8,
            temp_threshold=temp_threshold,
            random_opening_moves=random_opening_moves,
            resign_threshold=resign_threshold,
            resign_consecutive=resign_consecutive,
            resign_disabled_frac=resign_disabled_frac,
        )

    def play_game(self) -> Tuple[List[Tuple], GameStats]:
        """Play a single game (backward compat)."""
        examples, stats = self.worker.play_games(total_games=1)
        return examples, stats[0] if stats else GameStats()


def generate_self_play_data(model, num_games=100, num_simulations=800,
                            temp_threshold=30, random_opening_moves=0,
                            resign_threshold=-0.95) -> Tuple[List[Tuple], List[GameStats]]:
    """Generate training data using parallel self-play."""
    worker = ParallelSelfPlayWorker(
        model, num_simulations=num_simulations,
        num_parallel=min(8, num_games),
        temp_threshold=temp_threshold,
        random_opening_moves=random_opening_moves,
        resign_threshold=resign_threshold,
    )
    return worker.play_games(total_games=num_games)