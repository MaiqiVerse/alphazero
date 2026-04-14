"""
Batched MCTS with virtual loss for efficient GPU inference.

Key optimization: instead of calling model.predict() once per simulation,
we collect a BATCH of leaf nodes from parallel tree traversals, evaluate
them all in one GPU forward pass, then backpropagate all results.

This improves GPU utilization by 5-10x compared to sequential MCTS.

Architecture:
  1. Run `batch_size` parallel tree traversals using virtual loss
     to diversify paths
  2. Collect all leaf states into a single tensor
  3. One batched model.forward() call
  4. Backpropagate all results and remove virtual losses
  5. Repeat until total simulations reached
"""

import math
import numpy as np
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from game import AntichessGame, Move, ACTION_SIZE

# ─── Hyperparameters ───

C_PUCT = 1.5
DIR_ALPHA = 0.3
DIR_EPSILON = 0.25
VIRTUAL_LOSS = 3.0  # penalize in-flight nodes to encourage path diversity


class MCTSNode:
    """
    Tree node with support for virtual loss.

    Virtual loss: when a node is "in flight" (selected but not yet evaluated),
    we temporarily add a loss to discourage other parallel traversals from
    picking the same path. This ensures diverse leaf collection.
    """

    __slots__ = ['game', 'move', 'parent', 'children',
                 'visit_count', 'total_value', 'prior',
                 'is_expanded', 'virtual_loss_count']

    def __init__(self, game: AntichessGame, move: Optional[Move] = None,
                 parent: Optional['MCTSNode'] = None, prior: float = 0.0):
        self.game = game
        self.move = move
        self.parent = parent
        self.prior = prior
        self.children: List[MCTSNode] = []
        self.visit_count = 0
        self.total_value = 0.0
        self.is_expanded = False
        self.virtual_loss_count = 0  # number of in-flight traversals

    @property
    def q_value(self) -> float:
        """Q-value adjusted for virtual loss."""
        total_visits = self.visit_count + self.virtual_loss_count
        if total_visits == 0:
            return 0.0
        # Virtual losses count as losses (value = -1 each)
        adjusted_value = self.total_value - self.virtual_loss_count * VIRTUAL_LOSS
        return adjusted_value / total_visits

    def ucb_score(self) -> float:
        """PUCT score with virtual loss adjustment."""
        total_visits = self.visit_count + self.virtual_loss_count
        parent_visits = (self.parent.visit_count + self.parent.virtual_loss_count
                         if self.parent else 1)
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + total_visits)
        return self.q_value + u

    def add_virtual_loss(self):
        """Add virtual loss along the path from this node to root."""
        node = self
        while node is not None:
            node.virtual_loss_count += 1
            node = node.parent

    def remove_virtual_loss(self):
        """Remove virtual loss along the path from this node to root."""
        node = self
        while node is not None:
            node.virtual_loss_count -= 1
            node = node.parent


class BatchedMCTS:
    """
    MCTS with batched neural network evaluation.

    Instead of evaluating one leaf at a time (1 forward pass per simulation),
    we gather `batch_size` leaves per forward pass.

    For num_simulations=800 and batch_size=32:
      - Sequential: 800 forward passes, each with batch=1
      - Batched:     25 forward passes, each with batch=32
      -> ~25x fewer kernel launches, ~5-10x wall-clock speedup on GPU

    Args:
        model: neural network (must support batched forward pass)
        num_simulations: total number of MCTS simulations
        batch_size: number of leaves to collect per GPU batch
        device: torch device ('cuda' or 'cpu')
    """

    def __init__(self, model, num_simulations: int = 800,
                 batch_size: int = 32, device: torch.device = None):
        self.model = model
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def search(self, game: AntichessGame, temperature: float = 1.0,
               add_noise: bool = True) -> np.ndarray:
        """
        Run batched MCTS and return action probabilities.

        Algorithm:
          while simulations_done < num_simulations:
            1. Collect a batch of leaf nodes via parallel traversals
               with virtual loss to diversify paths
            2. Batch-evaluate all leaves with one model.forward()
            3. Expand leaves and backpropagate values
            4. Remove virtual losses

        Returns:
            pi: (ACTION_SIZE,) probability vector over actions
        """
        root = MCTSNode(game.clone())

        # Expand root with single evaluation
        self._expand_single(root)
        if not root.children:
            return np.zeros(ACTION_SIZE, dtype=np.float32)

        # Add Dirichlet noise to root for exploration
        if add_noise:
            noise = np.random.dirichlet([DIR_ALPHA] * len(root.children))
            for i, child in enumerate(root.children):
                child.prior = ((1 - DIR_EPSILON) * child.prior
                               + DIR_EPSILON * noise[i])

        # Main search loop: process in batches
        simulations_done = 0

        while simulations_done < self.num_simulations:
            # Determine batch size for this iteration
            remaining = self.num_simulations - simulations_done
            current_batch = min(self.batch_size, remaining)

            # ──────────────────────────────────────────────
            # Phase 1: COLLECT leaves via parallel traversals
            # ──────────────────────────────────────────────
            # Each traversal walks down the tree independently.
            # Virtual loss ensures they pick DIFFERENT paths.

            leaves = []          # leaf MCTSNode (need NN eval)
            leaf_states = []     # encoded states for batch inference
            leaf_terminals = []  # (node, value) for terminal nodes

            for _ in range(current_batch):
                # SELECT: traverse tree using PUCT
                node = root
                while node.is_expanded and node.children:
                    node = max(node.children, key=lambda c: c.ucb_score())

                # Apply virtual loss to this path so next traversal avoids it
                node.add_virtual_loss()

                # Check if terminal
                game_over, winner = node.game.is_game_over()
                if game_over:
                    if winner == -1:
                        value = 0.0
                    elif winner == game.turn:
                        value = 1.0
                    else:
                        value = -1.0
                    leaf_terminals.append((node, value))
                elif node.is_expanded:
                    # Already expanded (hit by earlier traversal in this batch)
                    # Use a neutral value; will be corrected over many sims
                    leaf_terminals.append((node, 0.0))
                else:
                    leaves.append(node)
                    leaf_states.append(node.game.encode())

            # ──────────────────────────────────────────────
            # Phase 2: BATCH EVALUATE all non-terminal leaves
            # ──────────────────────────────────────────────
            # THIS is the key: one GPU call for N states

            if leaf_states:
                policies, values = self._batch_predict(leaf_states)

                for i, node in enumerate(leaves):
                    policy = policies[i]
                    value = float(values[i])

                    # Expand the leaf node
                    self._expand_with_policy(node, policy)

                    # Backpropagate
                    self._backpropagate(node, value, game.turn)

                    # Remove virtual loss (path is now "settled")
                    node.remove_virtual_loss()

            # ──────────────────────────────────────────────
            # Phase 3: BACKPROP terminal nodes
            # ──────────────────────────────────────────────

            for node, value in leaf_terminals:
                self._backpropagate(node, value, game.turn)
                node.remove_virtual_loss()

            simulations_done += current_batch

        # Build action probabilities from visit counts
        return self._extract_policy(root, temperature)

    # ─── Neural Network Inference ───

    def _batch_predict(self, states: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch neural network inference.

        This is THE critical optimization:
          Before: 800 calls to model(tensor_of_shape_1x18x8x8)
          After:   25 calls to model(tensor_of_shape_32x18x8x8)

        GPU kernels have fixed launch overhead (~10-50μs each).
        Reducing 800 launches to 25 saves 8-40ms per move.
        More importantly, the GPU compute units are actually saturated
        with batch=32 whereas batch=1 leaves ~95% of cores idle.

        Args:
            states: list of N x (C, 8, 8) numpy arrays
        Returns:
            policies: (N, ACTION_SIZE) numpy array of softmax probabilities
            values: (N,) numpy array of value estimates
        """
        self.model.eval()

        with torch.no_grad():
            # Stack into single tensor: (N, C, 8, 8)
            batch_tensor = torch.FloatTensor(np.array(states)).to(self.device)

            # *** SINGLE forward pass for entire batch ***
            logits, v = self.model(batch_tensor)

            policies = F.softmax(logits, dim=1).cpu().numpy()
            values = v.squeeze(-1).cpu().numpy()

        return policies, values

    # ─── Tree Operations ───

    def _expand_single(self, node: MCTSNode):
        """Expand root node with a single NN evaluation."""
        if node.is_expanded:
            return

        node.is_expanded = True
        moves = node.game.legal_moves()
        if not moves:
            return

        state = node.game.encode()
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            logits, _ = self.model(x)
            policy = F.softmax(logits, dim=1).cpu().numpy()[0]

        self._create_children(node, moves, policy)

    def _expand_with_policy(self, node: MCTSNode, policy: np.ndarray):
        """Expand a leaf node using a pre-computed policy from batch eval."""
        if node.is_expanded:
            return

        node.is_expanded = True
        moves = node.game.legal_moves()
        if not moves:
            return

        self._create_children(node, moves, policy)

    def _create_children(self, node: MCTSNode, moves: List[Move],
                         policy: np.ndarray):
        """Create child nodes with priors extracted from policy vector."""
        legal_priors = []
        for move in moves:
            action_idx = move.to_action_index()
            legal_priors.append((move, policy[action_idx]))

        total = sum(p for _, p in legal_priors)
        if total < 1e-8:
            total = 1.0

        for move, prior in legal_priors:
            child_game = node.game.clone()
            child_game.apply_move(move)
            child = MCTSNode(child_game, move=move, parent=node,
                             prior=prior / total)
            node.children.append(child)

    def _backpropagate(self, node: MCTSNode, value: float, root_turn: int):
        """
        Backpropagate value from leaf to root, flipping sign at each level.

        The value from the NN is from the perspective of node.game.turn.
        As we walk up the tree, parent and child have opposite perspectives,
        so we alternate the sign.
        """
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
        """Convert root visit counts to action probability distribution."""
        pi = np.zeros(ACTION_SIZE, dtype=np.float32)
        for child in root.children:
            action_idx = child.move.to_action_index()
            pi[action_idx] = child.visit_count

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


# ─── Backward-compatible API ───

class MCTS(BatchedMCTS):
    """
    Drop-in replacement for the old sequential MCTS.

    Automatically picks batch_size based on device:
      - GPU: batch_size=32 (saturate GPU cores)
      - CPU: batch_size=8  (still helps reduce Python overhead)

    Usage is identical to before:
        mcts = MCTS(model, num_simulations=800)
        pi = mcts.search(game, temperature=1.0)
    """

    def __init__(self, model, num_simulations: int = 800,
                 batch_size: int = None, device: torch.device = None):
        device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        if batch_size is None:
            batch_size = 32 if device.type == 'cuda' else 8

        super().__init__(model, num_simulations, batch_size, device)
