"""
Arena: pit different AlphaZero checkpoints against each other and compute Elo ratings.

Usage:
    # Compare two specific checkpoints
    python arena.py --models ckpt_010.pt ckpt_020.pt --games 100

    # Round-robin tournament across all checkpoints in a folder
    python arena.py --dir checkpoints/ --games 40

    # Compare a checkpoint against a random baseline
    python arena.py --models ckpt_050.pt random --games 100

    # Quick test with fewer MCTS simulations
    python arena.py --models ckpt_010.pt ckpt_020.pt --games 20 --simulations 100
"""

import os
import glob
import math
import argparse
from itertools import combinations
from collections import defaultdict

import numpy as np
import torch

from game import AntichessGame, ACTION_SIZE
from model import AntichessNet
from mcts import MCTS


# ─── Elo Calculation ───

class EloTracker:
    """
    Track and compute Elo ratings from match results.
    Uses the standard Elo formula with K=32.
    """

    def __init__(self, initial_elo: float = 1500.0, k: float = 32.0):
        self.ratings = defaultdict(lambda: initial_elo)
        self.k = k
        self.history = []  # list of (player_a, player_b, result)

    def expected(self, ra: float, rb: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def update(self, player_a: str, player_b: str, result: float):
        """
        Update ratings after a game.
        result: 1.0 = player_a wins, 0.0 = player_b wins, 0.5 = draw
        """
        ra, rb = self.ratings[player_a], self.ratings[player_b]
        ea = self.expected(ra, rb)
        eb = self.expected(rb, ra)

        self.ratings[player_a] += self.k * (result - ea)
        self.ratings[player_b] += self.k * ((1.0 - result) - eb)
        self.history.append((player_a, player_b, result))

    def get_ratings(self) -> dict:
        return dict(sorted(self.ratings.items(), key=lambda x: -x[1]))

    def print_standings(self):
        print("\n" + "=" * 50)
        print(f"  {'Model':<30} {'Elo':>8}")
        print("=" * 50)
        for name, elo in self.get_ratings().items():
            print(f"  {name:<30} {elo:>8.1f}")
        print("=" * 50)


# ─── Random Player (baseline) ───

class RandomPlayer:
    """Baseline: picks random legal moves (no neural network)."""

    def predict(self, state):
        policy = np.ones(ACTION_SIZE, dtype=np.float32) / ACTION_SIZE
        return policy, 0.0


# ─── Match Logic ───

def load_model(path: str, device: torch.device) -> AntichessNet:
    """Load a model from checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    # Infer architecture from checkpoint
    config = checkpoint.get('config', {})
    model = AntichessNet(
        in_channels=config.get('in_channels', 18),
        num_res_blocks=config.get('num_res_blocks', 10),
        channels=config.get('channels', 128),
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def play_one_game(white_mcts: MCTS, black_mcts: MCTS,
                  max_moves: int = 200, verbose: bool = False) -> int:
    """
    Play a single game between two MCTS players.

    Returns:
        0 = white wins, 1 = black wins, -1 = draw
    """
    game = AntichessGame()
    move_num = 0

    while move_num < max_moves:
        over, winner = game.is_game_over()
        if over:
            if verbose:
                side = "White" if winner == 0 else ("Black" if winner == 1 else "Draw")
                print(f"  Game over at move {move_num}: {side}")
            return winner

        mcts = white_mcts if game.turn == 0 else black_mcts
        pi = mcts.search(game, temperature=0.1, add_noise=False)

        # Pick best action
        action = np.argmax(pi)
        legal = game.legal_moves()
        move = None
        for m in legal:
            if m.to_action_index() == action:
                move = m
                break
        if move is None:
            move = legal[0]

        if verbose and move_num < 10:
            print(f"  Move {move_num}: {'W' if game.turn==0 else 'B'} {move}")

        game.apply_move(move)
        move_num += 1

    return -1  # draw by move limit


def play_match(player_a, player_b, num_games: int,
               simulations: int, verbose: bool = False):
    """
    Play a match between two players. Each side plays both colors.

    Returns:
        (a_wins, b_wins, draws)
    """
    mcts_a = MCTS(player_a, num_simulations=simulations)
    mcts_b = MCTS(player_b, num_simulations=simulations)

    a_wins, b_wins, draws = 0, 0, 0

    for i in range(num_games):
        # Alternate colors
        if i % 2 == 0:
            # A plays white
            result = play_one_game(mcts_a, mcts_b, verbose=verbose and i < 2)
            if result == 0:
                a_wins += 1
            elif result == 1:
                b_wins += 1
            else:
                draws += 1
        else:
            # B plays white
            result = play_one_game(mcts_b, mcts_a, verbose=verbose and i < 2)
            if result == 0:
                b_wins += 1
            elif result == 1:
                a_wins += 1
            else:
                draws += 1

        total = i + 1
        if total % 10 == 0 or total == num_games:
            print(f"  Progress: {total}/{num_games}  "
                  f"A={a_wins} B={b_wins} Draw={draws}  "
                  f"A win%={a_wins/total:.1%}")

    return a_wins, b_wins, draws


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(
        description='Arena: compare AlphaZero checkpoints and compute Elo ratings')

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--models', nargs='+',
                       help='Paths to model checkpoints (or "random" for baseline)')
    group.add_argument('--dir',
                       help='Directory of checkpoints for round-robin tournament')

    parser.add_argument('--games', type=int, default=40,
                        help='Games per matchup (default: 40)')
    parser.add_argument('--simulations', type=int, default=400,
                        help='MCTS simulations per move (default: 400)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print individual game details')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Collect models
    model_paths = []
    if args.dir:
        model_paths = sorted(glob.glob(os.path.join(args.dir, '*.pt')))
        if not model_paths:
            print(f"No .pt files found in {args.dir}")
            return
        # Always include random baseline
        model_paths.append('random')
    else:
        model_paths = args.models

    print(f"\nModels: {model_paths}")
    print(f"Games per match: {args.games}")
    print(f"MCTS simulations: {args.simulations}\n")

    # Load models
    players = {}
    for path in model_paths:
        name = os.path.basename(path) if path != 'random' else 'random'
        if path == 'random':
            players[name] = RandomPlayer()
        else:
            print(f"Loading {name}...")
            players[name] = load_model(path, device)

    # Run matches
    elo = EloTracker()
    names = list(players.keys())

    if len(names) == 2:
        # Head-to-head
        a, b = names
        print(f"\n{'─'*50}")
        print(f"Match: {a}  vs  {b}")
        print(f"{'─'*50}")

        a_wins, b_wins, draws = play_match(
            players[a], players[b],
            num_games=args.games,
            simulations=args.simulations,
            verbose=args.verbose
        )

        print(f"\nResult: {a} wins {a_wins}, {b} wins {b_wins}, draws {draws}")
        win_rate = a_wins / args.games
        print(f"{a} win rate: {win_rate:.1%}")

        # Compute Elo difference from win rate
        if 0 < win_rate < 1:
            elo_diff = -400 * math.log10(1.0 / win_rate - 1.0)
            print(f"Estimated Elo difference: {elo_diff:+.0f}")

        # Update Elo for each game result
        for _ in range(a_wins):
            elo.update(a, b, 1.0)
        for _ in range(b_wins):
            elo.update(a, b, 0.0)
        for _ in range(draws):
            elo.update(a, b, 0.5)

    else:
        # Round-robin tournament
        matchups = list(combinations(names, 2))
        print(f"Round-robin: {len(matchups)} matchups\n")

        results_table = {}
        for i, (a, b) in enumerate(matchups):
            print(f"\n{'─'*50}")
            print(f"Match {i+1}/{len(matchups)}: {a}  vs  {b}")
            print(f"{'─'*50}")

            a_wins, b_wins, draws = play_match(
                players[a], players[b],
                num_games=args.games,
                simulations=args.simulations,
                verbose=args.verbose
            )

            results_table[(a, b)] = (a_wins, b_wins, draws)

            for _ in range(a_wins):
                elo.update(a, b, 1.0)
            for _ in range(b_wins):
                elo.update(a, b, 0.0)
            for _ in range(draws):
                elo.update(a, b, 0.5)

        # Print cross-table
        print("\n\nCross Table (wins/losses/draws):")
        print(f"{'':>25}", end='')
        for n in names:
            print(f"{n[:12]:>14}", end='')
        print()

        for a in names:
            print(f"{a[:24]:>25}", end='')
            for b in names:
                if a == b:
                    print(f"{'---':>14}", end='')
                elif (a, b) in results_table:
                    w, l, d = results_table[(a, b)]
                    print(f"{f'{w}/{l}/{d}':>14}", end='')
                elif (b, a) in results_table:
                    l, w, d = results_table[(b, a)]
                    print(f"{f'{w}/{l}/{d}':>14}", end='')
            print()

    # Final Elo standings
    elo.print_standings()

    # Print Elo progression chart (ASCII)
    if len(names) > 2:
        print("\nElo Progression (relative to random baseline):")
        ratings = elo.get_ratings()
        base = ratings.get('random', 1500)
        max_diff = max(abs(r - base) for r in ratings.values()) or 1

        for name, rating in ratings.items():
            diff = rating - base
            bar_len = int(40 * abs(diff) / max_diff)
            if diff >= 0:
                bar = ' ' * 20 + '█' * bar_len
            else:
                pad = 20 - bar_len
                bar = ' ' * pad + '█' * bar_len
            print(f"  {name[:20]:<20} {rating:7.1f} ({diff:+.0f}) {bar}")


if __name__ == '__main__':
    main()
