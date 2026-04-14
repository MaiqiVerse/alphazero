"""
Antichess (Suicide Chess) game logic.
Rules:
  - Captures are mandatory (if you can capture, you must)
  - First player to lose all pieces WINS
  - Stalemate (no legal moves) is also a WIN for that player
  - No castling; en passant IS supported (mandatory capture)
"""

import numpy as np
from typing import List, Tuple, Optional

# Piece encoding: 0=empty, 1-6=white P,N,B,R,Q,K, 7-12=black P,N,B,R,Q,K
EMPTY = 0
WP, WN, WB, WR, WQ, WK = 1, 2, 3, 4, 5, 6
BP, BN, BB, BR, BQ, BK = 7, 8, 9, 10, 11, 12

WHITE, BLACK = 0, 1

PIECE_TYPE = [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]  # maps piece -> type
PIECE_COLOR = [-1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # maps piece -> color

# Move encoding for neural network:
# We use a flat index: from_sq * 64 + to_sq, plus promotion info
# Total action space: 64*64 = 4096 base moves
# For promotions: 64*64 + from_col*4*4 = additional moves
# Simplified: we use 4096 + 4*8*4 = 4096+128 = 4224 action space
# (4 promo types x 8 columns x 4 directions for pawn promos)
# Actually let's keep it simple: action = from*64+to for non-promo,
# and for promos we encode promo_type in upper bits.
# Total action space = 64*64*5 = 20480 (5 = no_promo + Q/R/B/N promo)
# But that's large. Let's use a more compact scheme.
#
# Compact: 4672 actions
#   - 56 queen-like moves (8 directions x 7 distances) per square = 3584
#   - 8 knight moves per square = 512
#   - 9 underpromotions (3 piece types x 3 directions) per square on 2nd-to-last rank
#     (but this gets complex)
#
# Simplest practical approach: action = from_sq * 73 + move_type
# where move_type encodes direction+distance+promo
#
# For this implementation, we'll use the simplest: action = index in legal move list
# mapped to a fixed-size policy via a move-to-index mapping.
# We use: ACTION_SIZE = 4096 + 128 = 4224
#   action = from*64+to  for normal moves (0..4095)
#   action = 4096 + from_col*16 + dir*4 + promo_idx  for promotions
#     dir: 0=straight, 1=capture_left, 2=capture_right
#     promo_idx: 0=Q, 1=R, 2=B, 3=N

ACTION_SIZE = 4224

KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                (1, -2), (1, 2), (2, -1), (2, 1)]
KING_DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
             (0, 1), (1, -1), (1, 0), (1, 1)]
BISHOP_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ROOK_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUEEN_DIRS = BISHOP_DIRS + ROOK_DIRS


def rc(sq: int) -> Tuple[int, int]:
    return sq // 8, sq % 8


def idx(r: int, c: int) -> int:
    return r * 8 + c


def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < 8 and 0 <= c < 8


class Move:
    __slots__ = ['from_sq', 'to_sq', 'capture', 'promo', 'is_ep']

    def __init__(self, from_sq: int, to_sq: int, capture: bool = False,
                 promo: int = 0, is_ep: bool = False):
        self.from_sq = from_sq
        self.to_sq = to_sq
        self.capture = capture
        self.promo = promo  # 0 = no promotion, else piece type
        self.is_ep = is_ep  # en passant capture

    def to_action_index(self) -> int:
        """Convert move to action index for neural network."""
        if self.promo == 0:
            return self.from_sq * 64 + self.to_sq
        else:
            # Promotion move
            fr, fc = rc(self.from_sq)
            tr, tc = rc(self.to_sq)
            dc = tc - fc  # -1, 0, or 1
            direction = dc + 1  # 0=left, 1=straight, 2=right
            pt = PIECE_TYPE[self.promo]
            promo_idx = {5: 0, 4: 1, 3: 2, 2: 3}[pt]  # Q=0, R=1, B=2, N=3
            return 4096 + fc * 16 + direction * 4 + promo_idx

    def __repr__(self):
        cols = 'abcdefgh'
        s = f"{cols[self.from_sq%8]}{8-self.from_sq//8}{cols[self.to_sq%8]}{8-self.to_sq//8}"
        if self.promo:
            s += '=' + 'xPNBRQK'[PIECE_TYPE[self.promo]]
        return s


class AntichessGame:
    """
    Antichess game state.
    The board is stored from WHITE's perspective (row 0 = rank 8, row 7 = rank 1).
    """

    def __init__(self):
        self.board = np.zeros(64, dtype=np.int8)
        self._init_board()
        self.turn = WHITE
        self.move_count = 0
        self.en_passant_sq = -1  # square where EP capture can land, or -1
        self.history: List[int] = []  # board hashes for repetition detection

    def _init_board(self):
        # Black pieces on rows 0-1
        back = [BR, BN, BB, BQ, BK, BB, BN, BR]
        for i in range(8):
            self.board[i] = back[i]
            self.board[8 + i] = BP
            self.board[48 + i] = WP
        front = [WR, WN, WB, WQ, WK, WB, WN, WR]
        for i in range(8):
            self.board[56 + i] = front[i]

    def clone(self) -> 'AntichessGame':
        g = AntichessGame.__new__(AntichessGame)
        g.board = self.board.copy()
        g.turn = self.turn
        g.move_count = self.move_count
        g.en_passant_sq = self.en_passant_sq
        g.history = self.history.copy()
        return g

    def _pseudo_moves(self) -> List[Move]:
        """Generate all pseudo-legal moves for current side."""
        side = self.turn
        moves = []
        for sq in range(64):
            p = self.board[sq]
            if p == EMPTY or PIECE_COLOR[p] != side:
                continue
            r, c = rc(sq)
            pt = PIECE_TYPE[p]

            if pt == 1:  # Pawn
                d = -1 if side == WHITE else 1
                start_row = 6 if side == WHITE else 1
                promo_row = 0 if side == WHITE else 7
                fr = r + d

                # Forward
                if in_bounds(fr, c) and self.board[idx(fr, c)] == EMPTY:
                    if fr == promo_row:
                        for pp in ([WQ, WR, WB, WN] if side == WHITE else [BQ, BR, BB, BN]):
                            moves.append(Move(sq, idx(fr, c), False, pp))
                    else:
                        moves.append(Move(sq, idx(fr, c), False))
                        # Double push
                        if r == start_row and self.board[idx(r + 2 * d, c)] == EMPTY:
                            moves.append(Move(sq, idx(r + 2 * d, c), False))

                # Captures (including en passant)
                for dc in (-1, 1):
                    nc = c + dc
                    if not in_bounds(fr, nc):
                        continue
                    ti = idx(fr, nc)
                    tp = self.board[ti]
                    is_ep = (ti == self.en_passant_sq)
                    if (tp != EMPTY and PIECE_COLOR[tp] != side) or is_ep:
                        if fr == promo_row:
                            for pp in ([WQ, WR, WB, WN] if side == WHITE else [BQ, BR, BB, BN]):
                                moves.append(Move(sq, ti, True, pp))
                        else:
                            moves.append(Move(sq, ti, True, is_ep=is_ep))

            elif pt == 2:  # Knight
                for dr, dc in KNIGHT_MOVES:
                    nr, nc = r + dr, c + dc
                    if not in_bounds(nr, nc):
                        continue
                    ti = idx(nr, nc)
                    tp = self.board[ti]
                    if tp == EMPTY:
                        moves.append(Move(sq, ti, False))
                    elif PIECE_COLOR[tp] != side:
                        moves.append(Move(sq, ti, True))

            elif pt in (3, 4, 5):  # Bishop, Rook, Queen
                dirs = {3: BISHOP_DIRS, 4: ROOK_DIRS, 5: QUEEN_DIRS}[pt]
                for dr, dc in dirs:
                    nr, nc = r + dr, c + dc
                    while in_bounds(nr, nc):
                        ti = idx(nr, nc)
                        tp = self.board[ti]
                        if tp == EMPTY:
                            moves.append(Move(sq, ti, False))
                        elif PIECE_COLOR[tp] != side:
                            moves.append(Move(sq, ti, True))
                            break
                        else:
                            break
                        nr += dr
                        nc += dc

            elif pt == 6:  # King
                for dr, dc in KING_DIRS:
                    nr, nc = r + dr, c + dc
                    if not in_bounds(nr, nc):
                        continue
                    ti = idx(nr, nc)
                    tp = self.board[ti]
                    if tp == EMPTY:
                        moves.append(Move(sq, ti, False))
                    elif PIECE_COLOR[tp] != side:
                        moves.append(Move(sq, ti, True))

        return moves

    def legal_moves(self) -> List[Move]:
        """In antichess, if captures exist, you MUST capture."""
        all_moves = self._pseudo_moves()
        captures = [m for m in all_moves if m.capture]
        return captures if captures else all_moves

    def apply_move(self, move: Move):
        """Apply move in-place."""
        piece = self.board[move.from_sq]
        pt = PIECE_TYPE[piece]

        # En passant capture: remove the captured pawn
        if move.is_ep:
            # The captured pawn is on the same row as from_sq, same col as to_sq
            cap_sq = idx(rc(move.from_sq)[0], rc(move.to_sq)[1])
            self.board[cap_sq] = EMPTY

        self.board[move.to_sq] = move.promo if move.promo else piece
        self.board[move.from_sq] = EMPTY

        # Set en passant square if double pawn push
        self.en_passant_sq = -1
        if pt == 1:
            diff = move.to_sq - move.from_sq
            if abs(diff) == 16:  # double push
                # EP square is the square the pawn passed through
                self.en_passant_sq = (move.from_sq + move.to_sq) // 2

        self.turn = 1 - self.turn
        self.move_count += 1
        self.history.append(self._hash())

    def _hash(self) -> int:
        return hash((self.board.tobytes(), self.turn, self.en_passant_sq))

    def is_game_over(self) -> Tuple[bool, Optional[int]]:
        """
        Returns (is_over, winner).
        In antichess:
          - Lose all pieces -> you WIN
          - No legal moves (stalemate) -> you WIN
        winner: WHITE or BLACK or None (not over)
        """
        side = self.turn
        has_pieces = any(PIECE_COLOR[p] == side for p in self.board if p != EMPTY)
        if not has_pieces:
            return True, side  # Lost all pieces = win

        moves = self.legal_moves()
        if len(moves) == 0:
            return True, side  # Stalemate = win

        # Draw by repetition (simplified: 3-fold)
        if len(self.history) >= 8:
            h = self.history[-1]
            if self.history.count(h) >= 3:
                return True, -1  # Draw

        # Draw by move limit
        if self.move_count >= 200:
            return True, -1

        return False, None

    def get_result(self, player: int) -> float:
        """Get result from perspective of `player`. 1=win, 0=loss, 0.5=draw."""
        over, winner = self.is_game_over()
        if not over:
            return 0.5
        if winner == -1:
            return 0.5
        return 1.0 if winner == player else 0.0

    # ─── Neural Network Input Encoding ───

    def encode(self) -> np.ndarray:
        """
        Encode board state as tensor for neural network.
        Shape: (18, 8, 8)
          Channels 0-5:   current player's P, N, B, R, Q, K
          Channels 6-11:  opponent's P, N, B, R, Q, K
          Channel 12:     all ones if current player is white, else zeros
          Channel 13:     move count / 200 (normalized)
          Channel 14:     en passant square (1 at EP target, 0 elsewhere)
          Channels 15-17: reserved
        """
        planes = np.zeros((18, 8, 8), dtype=np.float32)
        side = self.turn
        opp = 1 - side

        for sq in range(64):
            r, c = rc(sq)
            p = self.board[sq]
            if p == EMPTY:
                continue
            pt = PIECE_TYPE[p]
            pc = PIECE_COLOR[p]
            if pc == side:
                planes[pt - 1, r, c] = 1.0
            else:
                planes[pt - 1 + 6, r, c] = 1.0

        if side == WHITE:
            planes[12, :, :] = 1.0
        planes[13, :, :] = self.move_count / 200.0

        if self.en_passant_sq >= 0:
            er, ec = rc(self.en_passant_sq)
            planes[14, er, ec] = 1.0

        return planes

    def get_symmetries(self, pi: np.ndarray, v: float):
        """
        Return symmetries of (state, policy, value) for data augmentation.
        For antichess we can mirror the board horizontally.
        """
        state = self.encode()
        # Original
        syms = [(state, pi, v)]

        # Horizontal flip
        flipped_state = np.flip(state, axis=2).copy()
        flipped_pi = pi.copy()
        # Flip action indices: swap columns
        new_pi = np.zeros_like(pi)
        for a in range(ACTION_SIZE):
            if a < 4096:
                fsq, tsq = a // 64, a % 64
                fr, fc = rc(fsq)
                tr, tc = rc(tsq)
                new_fsq = idx(fr, 7 - fc)
                new_tsq = idx(tr, 7 - tc)
                new_a = new_fsq * 64 + new_tsq
                new_pi[new_a] = pi[a]
            else:
                # Promo: flip column
                offset = a - 4096
                col = offset // 16
                rest = offset % 16
                new_col = 7 - col
                new_a = 4096 + new_col * 16 + rest
                new_pi[new_a] = pi[a]
        syms.append((flipped_state, new_pi, v))

        return syms
