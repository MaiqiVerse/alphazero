"""
Play against the trained AlphaZero model in the terminal.
Usage:
    python play.py                      # play with random initial weights
    python play.py --model checkpoint.pt  # play with trained model
"""

import argparse
import numpy as np

from game import AntichessGame, PIECE_COLOR, PIECE_TYPE, WHITE, BLACK, EMPTY
from model import AntichessNet
from mcts import MCTS

import torch

SYMBOLS = {
    1: '♙', 2: '♘', 3: '♗', 4: '♖', 5: '♕', 6: '♔',
    7: '♟', 8: '♞', 9: '♝', 10:'♜', 11:'♛', 12:'♚',
}


def print_board(game: AntichessGame):
    print("\n    a  b  c  d  e  f  g  h")
    print("  ┌──┬──┬──┬──┬──┬──┬──┬──┐")
    for r in range(8):
        row_str = f"{8-r} │"
        for c in range(8):
            p = game.board[r * 8 + c]
            if p == EMPTY:
                ch = '· '
            else:
                ch = SYMBOLS[p] + ' '
            row_str += ch + '│'
        print(row_str + f" {8-r}")
        if r < 7:
            print("  ├──┼──┼──┼──┼──┼──┼──┼──┤")
    print("  └──┴──┴──┴──┴──┴──┴──┴──┘")
    print("    a  b  c  d  e  f  g  h\n")


def parse_input(s: str, game: AntichessGame):
    """Parse user input like 'e2e4' or 'e7e8q' into a Move."""
    s = s.strip().lower()
    if len(s) < 4:
        return None
    cols = 'abcdefgh'
    try:
        fc = cols.index(s[0])
        fr = 8 - int(s[1])
        tc = cols.index(s[2])
        tr = 8 - int(s[3])
    except (ValueError, IndexError):
        return None

    promo_char = s[4] if len(s) > 4 else ''
    legal = game.legal_moves()
    for m in legal:
        mfr, mfc = m.from_sq // 8, m.from_sq % 8
        mtr, mtc = m.to_sq // 8, m.to_sq % 8
        if mfr == fr and mfc == fc and mtr == tr and mtc == tc:
            if m.promo:
                pt = PIECE_TYPE[m.promo]
                promo_map = {'q': 5, 'r': 4, 'b': 3, 'n': 2}
                if promo_char in promo_map and promo_map[promo_char] == pt:
                    return m
            else:
                if not promo_char:
                    return m
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--simulations', type=int, default=800)
    parser.add_argument('--color', choices=['white', 'black'], default='white')
    args = parser.parse_args()

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AntichessNet(num_res_blocks=10, channels=128).to(device)

    if args.model:
        ckpt = torch.load(args.model, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded model from {args.model}")
    else:
        print("Using randomly initialized model (untrained)")

    model.eval()
    mcts = MCTS(model, num_simulations=args.simulations)

    player_color = WHITE if args.color == 'white' else BLACK
    game = AntichessGame()

    print("═" * 40)
    print("  ANTICHESS (有子必吃棋) AlphaZero")
    print("  Must capture if possible.")
    print("  Lose all pieces to WIN!")
    print("  Enter moves like: e2e4, e7e8q")
    print("  Type 'quit' to exit.")
    print("═" * 40)

    while True:
        print_board(game)
        over, winner = game.is_game_over()
        if over:
            if winner == -1:
                print("Draw!")
            elif winner == player_color:
                print("You win!")
            else:
                print("AI wins!")
            break

        side_name = "White" if game.turn == WHITE else "Black"
        if game.turn == player_color:
            legal = game.legal_moves()
            captures = [m for m in legal if m.capture]
            if captures:
                print(f"[{side_name}] Must capture! Options: "
                      + ", ".join(str(m) for m in captures))
            else:
                print(f"[{side_name}] Legal moves: "
                      + ", ".join(str(m) for m in legal))

            while True:
                inp = input("Your move> ").strip()
                if inp == 'quit':
                    return
                move = parse_input(inp, game)
                if move:
                    break
                print("Invalid move. Try again.")
            game.apply_move(move)
        else:
            print(f"[{side_name}] AI thinking...")
            pi = mcts.search(game, temperature=0.1, add_noise=False)
            action = np.argmax(pi)
            legal = game.legal_moves()
            move = None
            for m in legal:
                if m.to_action_index() == action:
                    move = m
                    break
            if move is None:
                move = legal[0]
            print(f"AI plays: {move}")
            game.apply_move(move)


if __name__ == '__main__':
    main()
