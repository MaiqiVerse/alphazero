"""
UCI (Universal Chess Interface) adapter for Antichess AlphaZero.

This makes the engine compatible with lichess-bot and any UCI-compatible GUI.

Usage:
    python uci.py                           # uses default model path
    python uci.py --model checkpoints/best.pt --simulations 800

The engine announces itself as supporting the "antichess" / "giveaway" variant
via the UCI_Variant option.
"""

import sys
import os

# Force line-buffered stdout so UCI responses are sent immediately.
# reconfigure() is safe — it doesn't close/reopen the underlying fd.
sys.stdout.reconfigure(line_buffering=True)

import argparse
import threading
import time

import numpy as np
import torch

from game import AntichessGame, EMPTY, PIECE_COLOR, PIECE_TYPE, ACTION_SIZE
from game import WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK, WHITE, BLACK
from model import AntichessNet
from mcts import MCTS


# ─── FEN Parsing ───

PIECE_FROM_CHAR = {
    'P': WP, 'N': WN, 'B': WB, 'R': WR, 'Q': WQ, 'K': WK,
    'p': BP, 'n': BN, 'b': BB, 'r': BR, 'q': BQ, 'k': BK,
}

CHAR_FROM_PIECE = {v: k for k, v in PIECE_FROM_CHAR.items()}

STARTPOS_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1"


def fen_to_game(fen: str) -> AntichessGame:
    """Parse a FEN string into an AntichessGame state."""
    parts = fen.split()
    board_str = parts[0]
    turn_str = parts[1] if len(parts) > 1 else 'w'

    game = AntichessGame.__new__(AntichessGame)
    game.board = np.zeros(64, dtype=np.int8)
    game.history = []
    game.move_count = 0
    game.en_passant_sq = -1

    row, col = 0, 0
    for ch in board_str:
        if ch == '/':
            row += 1
            col = 0
        elif ch.isdigit():
            col += int(ch)
        else:
            game.board[row * 8 + col] = PIECE_FROM_CHAR.get(ch, EMPTY)
            col += 1

    game.turn = WHITE if turn_str == 'w' else BLACK

    # Parse en passant square from FEN (e.g. "e3" or "-")
    if len(parts) > 3 and parts[3] != '-':
        ep_str = parts[3]
        ep_col = 'abcdefgh'.index(ep_str[0])
        ep_row = 8 - int(ep_str[1])
        game.en_passant_sq = ep_row * 8 + ep_col

    if len(parts) > 5:
        game.move_count = int(parts[5]) * 2 + (1 if game.turn == BLACK else 0)

    return game


def game_to_fen(game: AntichessGame) -> str:
    """Convert game state to FEN string."""
    rows = []
    for r in range(8):
        row_str = ''
        empty_count = 0
        for c in range(8):
            p = game.board[r * 8 + c]
            if p == EMPTY:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += CHAR_FROM_PIECE.get(p, '?')
        if empty_count > 0:
            row_str += str(empty_count)
        rows.append(row_str)

    turn = 'w' if game.turn == WHITE else 'b'
    return f"{'/'.join(rows)} {turn} - - 0 {game.move_count // 2 + 1}"


# ─── Move Parsing (UCI long algebraic notation) ───

PROMO_CHAR_TO_TYPE = {
    'q': (WQ, BQ), 'r': (WR, BR), 'b': (WB, BB), 'n': (WN, BN),
}

PROMO_TYPE_TO_CHAR = {
    WQ: 'q', WR: 'r', WB: 'b', WN: 'n',
    BQ: 'q', BR: 'r', BB: 'b', BN: 'n',
}


def uci_to_move(uci_str: str, game: AntichessGame):
    """
    Parse UCI move string (e.g. 'e2e4', 'e7e8q') and find matching legal move.
    """
    uci_str = uci_str.strip().lower()
    cols = 'abcdefgh'

    from_col = cols.index(uci_str[0])
    from_row = 8 - int(uci_str[1])
    to_col = cols.index(uci_str[2])
    to_row = 8 - int(uci_str[3])
    promo_char = uci_str[4] if len(uci_str) > 4 else ''

    from_sq = from_row * 8 + from_col
    to_sq = to_row * 8 + to_col

    for m in game.legal_moves():
        if m.from_sq == from_sq and m.to_sq == to_sq:
            if promo_char:
                if m.promo and PROMO_TYPE_TO_CHAR.get(m.promo) == promo_char:
                    return m
            else:
                if not m.promo:
                    return m
    return None


def move_to_uci(move) -> str:
    """Convert a Move to UCI string."""
    cols = 'abcdefgh'
    s = (f"{cols[move.from_sq % 8]}{8 - move.from_sq // 8}"
         f"{cols[move.to_sq % 8]}{8 - move.to_sq // 8}")
    if move.promo:
        s += PROMO_TYPE_TO_CHAR.get(move.promo, '')
    return s


# ─── UCI Engine ───

class UCIEngine:
    """
    UCI protocol handler for Antichess AlphaZero.
    """

    def __init__(self, model_path: str = None, simulations: int = 800):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.simulations = simulations
        self.model_path = model_path

        # Game state
        self.game = AntichessGame()

        # Search control
        self.searching = False
        self.stop_flag = False

        # Start loading model in background thread so UCI handshake
        # can complete immediately (lichess-bot has a 60s timeout).
        self.model = None
        self.mcts = None
        self._model_ready = threading.Event()
        self._load_thread = threading.Thread(target=self._load_model, daemon=True)
        self._load_thread.start()

    def _wait_for_model(self):
        """Block until model is loaded. Called before any search."""
        if not self._model_ready.is_set():
            print("Waiting for model...", file=sys.stderr, flush=True)
            self._model_ready.wait()


    def _load_model(self):
        """Load neural network model. Runs in background thread.
        MUST NOT write to stdout — that would corrupt the UCI protocol."""
        self.model = AntichessNet(
            in_channels=18,
            num_res_blocks=10,
            channels=128,
        ).to(self.device)

        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device,
                                    weights_only=False)
            if 'model_state_dict' in checkpoint:
                config = checkpoint.get('config', {})
                self.model = AntichessNet(
                    in_channels=config.get('in_channels', 18),
                    num_res_blocks=config.get('num_res_blocks', 10),
                    channels=config.get('channels', 128),
                ).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            # stderr only — stdout is reserved for UCI protocol
            print(f"Model loaded: {self.model_path}", file=sys.stderr, flush=True)
        else:
            print("WARNING: No model, using random weights", file=sys.stderr, flush=True)

        self.model.eval()
        self.mcts = MCTS(self.model, num_simulations=self.simulations)
        self._model_ready.set()

    def handle_uci(self):
        """Respond to 'uci' command."""
        send("id name AntichessAlphaZero")
        send("id author AlphaZero-Antichess")
        send("")
        # Options
        send("option name UCI_Variant type combo default antichess var antichess var giveaway var suicide")
        send("option name Simulations type spin default 800 min 50 max 10000")
        send("option name ModelPath type string default <empty>")
        send("uciok")

    def handle_setoption(self, tokens: list):
        """Handle 'setoption name X value Y'."""
        try:
            name_idx = tokens.index('name') + 1
            value_idx = tokens.index('value') + 1
        except ValueError:
            return

        name = tokens[name_idx].lower()
        value = ' '.join(tokens[value_idx:])

        if name == 'simulations':
            self.simulations = int(value)
            if self.mcts:
                self.mcts.num_simulations = self.simulations
            log(f"info string Set simulations to {self.simulations}")
        elif name == 'modelpath':
            if value and value != '<empty>':
                self.model_path = value
                self._load_model()
                log(f"info string Loaded model: {value}")
        elif name == 'uci_variant':
            log(f"info string Variant set to {value}")

    def handle_isready(self):
        """Handle 'isready' command. Wait for model if still loading."""
        self._wait_for_model()
        send("readyok")

    def handle_ucinewgame(self):
        """Handle 'ucinewgame' command."""
        self.game = AntichessGame()

    def handle_position(self, tokens: list):
        """
        Handle 'position' command.
        Formats:
            position startpos
            position startpos moves e2e4 e7e5 ...
            position fen <fen> moves ...
        """
        idx = 0

        if 'startpos' in tokens:
            self.game = AntichessGame()
            idx = tokens.index('startpos') + 1
        elif 'fen' in tokens:
            fen_idx = tokens.index('fen') + 1
            # FEN has 6 parts
            fen_parts = tokens[fen_idx:fen_idx + 6]
            fen = ' '.join(fen_parts)
            self.game = fen_to_game(fen)
            idx = fen_idx + 6
        else:
            return

        # Apply moves
        if 'moves' in tokens:
            moves_idx = tokens.index('moves') + 1
            for move_str in tokens[moves_idx:]:
                move = uci_to_move(move_str, self.game)
                if move:
                    self.game.apply_move(move)
                else:
                    log(f"info string WARNING: illegal move {move_str}")

    def handle_go(self, tokens: list):
        """
        Handle 'go' command. Start searching for best move.

        Supported go parameters:
            movetime <ms>   - search for exactly this many milliseconds
            wtime/btime     - remaining time on clock
            winc/binc       - increment per move
            depth <n>       - search to depth n (mapped to simulation count)
            nodes <n>       - limit total nodes (mapped to simulations)
            infinite        - search until 'stop'
        """
        # Parse time control
        params = {}
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in ('wtime', 'btime', 'winc', 'binc', 'movetime',
                     'depth', 'nodes', 'movestogo'):
                if i + 1 < len(tokens):
                    params[t] = int(tokens[i + 1])
                    i += 2
                    continue
            elif t == 'infinite':
                params['infinite'] = True
            i += 1

        # Determine number of simulations based on time control
        sims = self.simulations

        if 'movetime' in params:
            # Rough estimate: ~1ms per simulation on GPU, ~5ms on CPU
            ms_per_sim = 1 if self.device.type == 'cuda' else 5
            sims = max(50, params['movetime'] // ms_per_sim)
        elif 'wtime' in params or 'btime' in params:
            # Use a fraction of remaining time
            if self.game.turn == WHITE:
                remaining = params.get('wtime', 60000)
                inc = params.get('winc', 0)
            else:
                remaining = params.get('btime', 60000)
                inc = params.get('binc', 0)

            moves_left = max(20, 60 - self.game.move_count // 2)
            think_time = remaining / moves_left + inc * 0.8
            think_time = min(think_time, remaining * 0.5)  # never use more than half
            think_time = max(think_time, 100)  # at least 100ms

            ms_per_sim = 1 if self.device.type == 'cuda' else 5
            sims = max(50, int(think_time / ms_per_sim))
        elif 'depth' in params:
            # Map depth to simulations (rough heuristic)
            sims = params['depth'] * 100
        elif 'nodes' in params:
            sims = params['nodes']

        sims = min(sims, 10000)  # cap

        # Run search in background thread
        self.stop_flag = False
        self.searching = True

        def search_thread():
            try:
                self._wait_for_model()
                self.mcts.num_simulations = sims
                log(f"info string Searching with {sims} simulations...")

                start_time = time.time()
                pi = self.mcts.search(self.game, temperature=0.1, add_noise=False)
                elapsed = time.time() - start_time

                # Find best move
                action = np.argmax(pi)
                legal = self.game.legal_moves()
                best_move = None
                for m in legal:
                    if m.to_action_index() == action:
                        best_move = m
                        break
                if best_move is None and legal:
                    best_move = legal[0]

                if best_move is None:
                    send("bestmove 0000")
                else:
                    uci_move = move_to_uci(best_move)

                    # Send info
                    nodes = sims
                    nps = int(nodes / elapsed) if elapsed > 0 else 0
                    elapsed_ms = int(elapsed * 1000)

                    # Get value estimate
                    state = self.game.encode()
                    _, value = self.model.predict(state)
                    cp = int(value * 100)  # convert to centipawns

                    send(f"info depth 1 seldepth 1 score cp {cp} "
                         f"nodes {nodes} nps {nps} time {elapsed_ms} "
                         f"pv {uci_move}")
                    send(f"bestmove {uci_move}")

            except Exception as e:
                log(f"info string Error in search: {e}")
                legal = self.game.legal_moves()
                if legal:
                    send(f"bestmove {move_to_uci(legal[0])}")
                else:
                    send("bestmove 0000")
            finally:
                self.searching = False

        thread = threading.Thread(target=search_thread, daemon=True)
        thread.start()

    def handle_stop(self):
        """Handle 'stop' command."""
        self.stop_flag = True

    def handle_quit(self):
        """Handle 'quit' command."""
        sys.exit(0)


# ─── I/O ───

def send(msg: str):
    """Send a message to the GUI (stdout)."""
    sys.stdout.write(msg + '\n')
    sys.stdout.flush()


def log(msg: str):
    """Send debug info (also via stdout as 'info string')."""
    send(msg)


# ─── Main Loop ───

def main():
    parser = argparse.ArgumentParser(description='Antichess AlphaZero UCI Engine')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--simulations', type=int, default=800,
                        help='Default MCTS simulations per move')
    args, _ = parser.parse_known_args()

    engine = UCIEngine(model_path=args.model, simulations=args.simulations)

    while True:
        try:
            line = input().strip()
        except EOFError:
            break

        if not line:
            continue

        tokens = line.split()
        cmd = tokens[0]

        if cmd == 'uci':
            engine.handle_uci()
        elif cmd == 'setoption':
            engine.handle_setoption(tokens)
        elif cmd == 'isready':
            engine.handle_isready()
        elif cmd == 'ucinewgame':
            engine.handle_ucinewgame()
        elif cmd == 'position':
            engine.handle_position(tokens)
        elif cmd == 'go':
            engine.handle_go(tokens[1:])
        elif cmd == 'stop':
            engine.handle_stop()
        elif cmd == 'quit':
            engine.handle_quit()
        elif cmd == 'd':
            # Debug: print board
            send(game_to_fen(engine.game))
        else:
            log(f"info string Unknown command: {cmd}")


if __name__ == '__main__':
    main()