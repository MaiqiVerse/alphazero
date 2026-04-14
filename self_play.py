"""
Self-play module with:
  - Opening diversification (random first N moves)
  - Early resignation (when value head says game is lost)
  - Game statistics for logging
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from game import AntichessGame, ACTION_SIZE
from mcts import MCTS


@dataclass
class GameStats:
    """Statistics from a single self-play game."""
    num_moves: int = 0
    resigned: bool = False
    winner: int = -1  # 0=white, 1=black, -1=draw


class SelfPlayWorker:
    """
    Generates training data through self-play.

    Features:
      - Opening diversification: first `random_opening_moves` are random
        legal moves to avoid the network collapsing to one opening line.
      - Resignation: if value head output < `resign_threshold` for
        `resign_consecutive` moves in a row, the game is resigned.
        A fraction of games (`resign_disabled_frac`) disable resignation
        to avoid blind spots where the network overestimates losing positions.
    """

    def __init__(self, model, num_simulations: int = 800,
                 temp_threshold: int = 30,
                 random_opening_moves: int = 0,
                 resign_threshold: float = -0.95,
                 resign_consecutive: int = 5,
                 resign_disabled_frac: float = 0.1):
        self.mcts = MCTS(model, num_simulations)
        self.temp_threshold = temp_threshold
        self.random_opening_moves = random_opening_moves
        self.resign_threshold = resign_threshold
        self.resign_consecutive = resign_consecutive
        self.resign_disabled_frac = resign_disabled_frac

    def play_game(self) -> Tuple[List[Tuple[np.ndarray, np.ndarray, float]], GameStats]:
        """
        Play one full game of self-play.

        Returns:
            examples: list of (state, pi, value) training tuples
            stats: GameStats for logging
        """
        game = AntichessGame()
        trajectory = []  # (state, pi, current_player)
        stats = GameStats()

        # Decide if resignation is enabled for this game
        resign_enabled = np.random.random() > self.resign_disabled_frac
        consecutive_low = [0, 0]  # per-side counter

        move_num = 0
        resigned_side = None

        while True:
            over, winner = game.is_game_over()
            if over:
                break

            current_side = game.turn
            legal = game.legal_moves()

            # ── Opening diversification ──
            if move_num < self.random_opening_moves:
                # Random legal move, no MCTS
                move = legal[np.random.randint(len(legal))]
                # Still record state/policy for training
                state = game.encode()
                pi = np.zeros(ACTION_SIZE, dtype=np.float32)
                pi[move.to_action_index()] = 1.0
                trajectory.append((state, pi, current_side))
                game.apply_move(move)
                move_num += 1
                continue

            # ── MCTS search ──
            temperature = 1.0 if move_num < self.temp_threshold else 0.1
            pi = self.mcts.search(game, temperature=temperature, add_noise=True)

            # Record state and policy
            state = game.encode()
            trajectory.append((state, pi, current_side))

            # ── Resignation check ──
            if resign_enabled and move_num >= self.random_opening_moves + 10:
                # Get value estimate from MCTS root (already computed)
                _, value = self.mcts.model.predict(state)
                if value < self.resign_threshold:
                    consecutive_low[current_side] += 1
                else:
                    consecutive_low[current_side] = 0

                if consecutive_low[current_side] >= self.resign_consecutive:
                    # This side resigns → opponent wins
                    # In antichess, "losing" means having more pieces, so
                    # resignation means "I can't get rid of my pieces"
                    resigned_side = current_side
                    stats.resigned = True
                    break

            # Sample action from policy
            action = np.random.choice(ACTION_SIZE, p=pi)

            # Find corresponding move
            move = None
            for m in legal:
                if m.to_action_index() == action:
                    move = m
                    break
            if move is None:
                move = legal[np.random.randint(len(legal))]

            game.apply_move(move)
            move_num += 1

        # Determine winner
        if resigned_side is not None:
            # The other side wins (in antichess, the resigner had too many pieces)
            winner = 1 - resigned_side
        else:
            _, winner = game.is_game_over()

        stats.num_moves = move_num
        stats.winner = winner if winner is not None else -1

        # Assign values and build examples with augmentation
        examples = []
        for state, pi, player in trajectory:
            if winner == -1:
                value = 0.0
            elif winner == player:
                value = 1.0
            else:
                value = -1.0
            examples.append((state, pi, value))

            # Data augmentation: horizontal mirror
            flipped_state = np.flip(state, axis=2).copy()
            flipped_pi = _flip_policy(pi)
            examples.append((flipped_state, flipped_pi, value))

        return examples, stats


def _flip_policy(pi: np.ndarray) -> np.ndarray:
    """Horizontally flip a policy vector."""
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


def generate_self_play_data(model, num_games: int = 100,
                            num_simulations: int = 800,
                            temp_threshold: int = 30,
                            random_opening_moves: int = 0,
                            resign_threshold: float = -0.95) -> Tuple[List[Tuple], List[GameStats]]:
    """
    Generate training data from multiple self-play games.

    Returns:
        all_examples: combined (state, pi, value) tuples
        all_stats: per-game statistics
    """
    worker = SelfPlayWorker(
        model, num_simulations, temp_threshold,
        random_opening_moves=random_opening_moves,
        resign_threshold=resign_threshold,
    )
    all_examples = []
    all_stats = []

    for i in range(num_games):
        examples, stats = worker.play_game()
        all_examples.extend(examples)
        all_stats.append(stats)
        if (i + 1) % 10 == 0 or (i + 1) == num_games:
            avg_len = np.mean([s.num_moves for s in all_stats])
            resigned = sum(1 for s in all_stats if s.resigned)
            print(f"Self-play {i+1}/{num_games}, "
                  f"avg_len={avg_len:.0f}, resigned={resigned}, "
                  f"examples={len(all_examples)}")

    return all_examples, all_stats
