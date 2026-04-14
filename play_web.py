"""
Local web UI for playing Antichess against your AlphaZero model.

Usage:
    python play_web.py --model checkpoints/best.pt --simulations 800
    # Then open http://localhost:8080 in your browser

Dependencies: only Python stdlib + torch + numpy (no Flask/Django needed)
"""

import os
import sys
import json
import argparse
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import torch

from game import (AntichessGame, EMPTY, PIECE_COLOR, PIECE_TYPE,
                  WP, WN, WB, WR, WQ, WK, BP, BN, BB, BR, BQ, BK,
                  WHITE, BLACK, ACTION_SIZE)
from model import AntichessNet
from mcts import MCTS

# ─── Globals ───
engine_model = None
engine_mcts = None
game_state = None
player_color = WHITE


PIECE_SYMBOLS = {
    WK: '♔', WQ: '♕', WR: '♖', WB: '♗', WN: '♘', WP: '♙',
    BK: '♚', BQ: '♛', BR: '♜', BB: '♝', BN: '♞', BP: '♟',
}


def get_board_json(game: AntichessGame) -> dict:
    """Serialize game state to JSON for the frontend."""
    board = []
    for sq in range(64):
        p = game.board[sq]
        board.append({
            'piece': int(p),
            'symbol': PIECE_SYMBOLS.get(p, ''),
            'color': int(PIECE_COLOR[p]) if p != EMPTY else -1,
        })

    legal = game.legal_moves()
    moves = []
    for m in legal:
        moves.append({
            'from': int(m.from_sq),
            'to': int(m.to_sq),
            'capture': m.capture,
            'promo': int(m.promo),
        })

    over, winner = game.is_game_over()

    return {
        'board': board,
        'turn': int(game.turn),
        'playerColor': int(player_color),
        'legalMoves': moves,
        'gameOver': over,
        'winner': int(winner) if winner is not None else None,
        'moveCount': game.move_count,
    }


def apply_player_move(from_sq: int, to_sq: int, promo: int = 0) -> dict:
    """Apply player's move and get AI response."""
    global game_state

    # Find matching legal move
    legal = game_state.legal_moves()
    move = None
    for m in legal:
        if m.from_sq == from_sq and m.to_sq == to_sq:
            if promo:
                if m.promo == promo:
                    move = m
                    break
            else:
                if not m.promo:
                    move = m
                    break
    # If promo not specified but move requires it, default to queen
    if move is None and promo == 0:
        for m in legal:
            if m.from_sq == from_sq and m.to_sq == to_sq and m.promo:
                pt = PIECE_TYPE[m.promo]
                if pt == 5:  # Queen
                    move = m
                    break

    if move is None:
        return {'error': 'Illegal move'}

    game_state.apply_move(move)
    result = get_board_json(game_state)

    # Check if game over after player move
    if result['gameOver']:
        return result

    # AI responds
    if game_state.turn != player_color:
        ai_move = ai_think()
        if ai_move:
            result['aiMove'] = {'from': ai_move.from_sq, 'to': ai_move.to_sq}
            result = get_board_json(game_state)
            result['aiMove'] = {'from': ai_move.from_sq, 'to': ai_move.to_sq}

    return result


def ai_think():
    """Run MCTS and apply AI move."""
    global game_state

    over, _ = game_state.is_game_over()
    if over:
        return None

    pi = engine_mcts.search(game_state, temperature=0.1, add_noise=False)
    action = np.argmax(pi)

    legal = game_state.legal_moves()
    move = None
    for m in legal:
        if m.to_action_index() == action:
            move = m
            break
    if move is None and legal:
        move = legal[0]

    if move:
        game_state.apply_move(move)

    return move


# ─── HTML UI ───

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Antichess AlphaZero</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'Noto Sans SC', sans-serif;
    background: #1a1714;
    color: #e8dcc8;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    padding: 20px;
  }

  h1 {
    font-size: 26px;
    font-weight: 700;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #c9a84c, #e8dcc8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
  }

  .subtitle { font-size: 13px; color: #8b7d6b; margin-bottom: 16px; }

  .controls {
    display: flex; gap: 10px; margin-bottom: 14px; flex-wrap: wrap;
    justify-content: center;
  }

  .controls button, .controls select {
    padding: 7px 16px; font-size: 13px; font-weight: 700;
    background: #252220; color: #c9a84c;
    border: 1px solid #997a2d; border-radius: 6px;
    cursor: pointer; letter-spacing: 1px;
    transition: background 0.2s;
  }

  .controls button:hover { background: #333028; }

  .status {
    font-size: 15px; margin: 10px 0; min-height: 24px; text-align: center;
  }

  .status.thinking { color: #c9a84c; animation: pulse 1.2s infinite; }
  .status.gameover { color: #e8c840; font-weight: 700; font-size: 18px; }

  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.4} }

  #board-container {
    position: relative;
    width: min(88vw, 480px); height: min(88vw, 480px);
    border-radius: 4px; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.6);
    border: 2px solid #997a2d;
  }

  #board {
    display: grid;
    grid-template-columns: repeat(8, 1fr);
    grid-template-rows: repeat(8, 1fr);
    width: 100%; height: 100%;
  }

  .sq {
    display: flex; align-items: center; justify-content: center;
    position: relative; cursor: default;
    font-size: min(7vw, 42px); user-select: none;
    transition: background 0.12s;
  }

  .sq.light { background: #f0d9b5; }
  .sq.dark { background: #b58863; }
  .sq.selected { background: #e8c840 !important; }
  .sq.last-move { background: #d4b44a !important; }
  .sq.clickable { cursor: pointer; }

  .sq .dot {
    width: 14px; height: 14px; border-radius: 50%;
    background: rgba(0,0,0,0.18); position: absolute;
  }

  .sq .capture-ring {
    position: absolute; inset: 4px;
    border: 4px solid rgba(0,0,0,0.22); border-radius: 50%;
  }

  .sq .coord {
    position: absolute; font-size: 10px; font-weight: 700; opacity: 0.6;
  }

  .sq .coord-row { top: 2px; left: 3px; }
  .sq .coord-col { bottom: 1px; right: 3px; }

  .sq.light .coord { color: #b58863; }
  .sq.dark .coord { color: #f0d9b5; }

  .piece { z-index: 1; line-height: 1; transition: transform 0.12s; }
  .sq.selected .piece { transform: scale(1.15); filter: drop-shadow(0 0 8px rgba(201,168,76,0.8)); }

  .captured {
    display: flex; align-items: center; gap: 2px;
    min-height: 28px; flex-wrap: wrap; margin: 4px 0;
  }

  .captured .label { font-size: 11px; color: #8b7d6b; margin-right: 6px; }
  .captured span:not(.label) { font-size: 18px; opacity: 0.65; }

  #promo-modal {
    display: none; position: absolute; inset: 0;
    background: rgba(0,0,0,0.55);
    align-items: center; justify-content: center; z-index: 10;
  }

  #promo-modal.active { display: flex; }

  #promo-choices {
    display: flex; gap: 8px; padding: 16px;
    background: #252220; border-radius: 8px;
    border: 1px solid #c9a84c;
  }

  #promo-choices button {
    font-size: 36px; padding: 8px 14px;
    background: none; border: 1px solid #8b7355;
    border-radius: 6px; cursor: pointer; color: #e8dcc8;
    transition: background 0.2s;
  }

  #promo-choices button:hover { background: #333028; }

  .move-log {
    margin-top: 10px; max-width: 480px; width: 100%;
    max-height: 80px; overflow-y: auto;
    padding: 6px 10px; background: #252220; border-radius: 6px;
    font-size: 12px; color: #8b7d6b; font-family: monospace;
    display: flex; flex-wrap: wrap; gap: 2px 8px;
  }
</style>
</head>
<body>
  <h1>♚ ANTICHESS ALPHAZERO</h1>
  <p class="subtitle">有子必吃 · 失去所有棋子者获胜</p>

  <div class="controls">
    <button onclick="newGame(0)">执白</button>
    <button onclick="newGame(1)">执黑</button>
    <button onclick="undoMove()">悔棋</button>
  </div>

  <div class="captured" id="captured-top"></div>

  <div id="board-container">
    <div id="board"></div>
    <div id="promo-modal">
      <div id="promo-choices"></div>
    </div>
  </div>

  <div class="captured" id="captured-bottom"></div>
  <div class="status" id="status">加载中...</div>
  <div class="move-log" id="move-log"></div>

<script>
let state = null;
let selected = null;
let validTargets = [];
let lastMove = null;
let flipped = false;
let moveLog = [];
let capturedW = [], capturedB = [];
const SYM = {1:'♙',2:'♘',3:'♗',4:'♖',5:'♕',6:'♔',7:'♟',8:'♞',9:'♝',10:'♜',11:'♛',12:'♚'};
const COLS = 'abcdefgh';

async function api(endpoint, params={}) {
  const url = endpoint + '?' + new URLSearchParams(params);
  const res = await fetch(url);
  return res.json();
}

async function newGame(color) {
  capturedW = []; capturedB = []; moveLog = [];
  flipped = color === 1;
  state = await api('/api/new', {color});
  selected = null; lastMove = null;
  render();

  // If AI goes first
  if (state.turn !== state.playerColor && !state.gameOver) {
    setStatus('thinking');
    state = await api('/api/ai_move');
    if (state.aiMove) lastMove = state.aiMove;
    render();
  }
}

async function undoMove() {
  state = await api('/api/undo');
  lastMove = null; selected = null;
  render();
}

function setStatus(type, text) {
  const el = document.getElementById('status');
  el.className = 'status';
  if (type === 'thinking') {
    el.classList.add('thinking');
    el.textContent = '🤔 AI 思考中...';
  } else if (type === 'gameover') {
    el.classList.add('gameover');
    el.textContent = text;
  } else {
    el.textContent = text || '';
  }
}

async function clickSquare(sq) {
  if (!state || state.gameOver) return;
  if (state.turn !== state.playerColor) return;

  const p = state.board[sq];

  if (selected !== null) {
    // Try to make a move
    const move = state.legalMoves.find(m => m.from === selected && m.to === sq && !m.promo);
    const promoMoves = state.legalMoves.filter(m => m.from === selected && m.to === sq && m.promo);

    if (promoMoves.length > 0) {
      showPromoModal(selected, sq, promoMoves);
      return;
    }

    if (move) {
      await makeMove(move.from, move.to, 0);
      return;
    }
  }

  // Select a piece
  if (p.piece !== 0 && p.color === state.playerColor) {
    selected = sq;
    validTargets = state.legalMoves.filter(m => m.from === sq).map(m => m.to);
    renderBoard();
  } else {
    selected = null;
    validTargets = [];
    renderBoard();
  }
}

function showPromoModal(from, to, moves) {
  const modal = document.getElementById('promo-modal');
  const choices = document.getElementById('promo-choices');
  choices.innerHTML = '';
  const side = state.playerColor;

  // Deduplicate by promo piece type
  const seen = new Set();
  for (const m of moves) {
    if (seen.has(m.promo)) continue;
    seen.add(m.promo);
    const btn = document.createElement('button');
    btn.textContent = SYM[m.promo];
    btn.onclick = async () => {
      modal.classList.remove('active');
      await makeMove(from, to, m.promo);
    };
    choices.appendChild(btn);
  }
  modal.classList.add('active');
}

async function makeMove(from, to, promo) {
  const captured = state.board[to].piece;
  if (captured) {
    if (state.board[to].color === 0) capturedW.push(captured);
    else capturedB.push(captured);
  }

  const fromStr = COLS[from%8] + (8 - Math.floor(from/8));
  const toStr = COLS[to%8] + (8 - Math.floor(to/8));
  moveLog.push((captured ? fromStr+'x'+toStr : fromStr+'-'+toStr));

  setStatus('thinking');
  selected = null; validTargets = [];

  state = await api('/api/move', {from, to, promo});

  if (state.error) {
    setStatus('', '非法走法');
    return;
  }

  lastMove = {from, to};

  if (state.aiMove) {
    // Record AI capture
    const aiCaptured = state.board[state.aiMove.to]; // already applied
    // We rely on server state; just record move text
    const af = COLS[state.aiMove.from%8] + (8-Math.floor(state.aiMove.from/8));
    const at = COLS[state.aiMove.to%8] + (8-Math.floor(state.aiMove.to/8));
    moveLog.push(af+'-'+at);
    lastMove = state.aiMove;
  }

  render();
}

function render() {
  renderBoard();
  renderCaptured();
  renderMoveLog();

  if (!state) {
    setStatus('', '点击「执白」或「执黑」开始');
    return;
  }

  if (state.gameOver) {
    const w = state.winner;
    if (w === null || w === -1) setStatus('gameover', '和棋！');
    else if (w === state.playerColor) setStatus('gameover', '🎉 你赢了！');
    else setStatus('gameover', '💀 AI 赢了！');
  } else {
    const who = state.turn === state.playerColor ? '你' : 'AI';
    setStatus('', (state.turn === 0 ? '白方' : '黑方') + '走棋 (' + who + ')');
  }
}

function renderBoard() {
  const boardEl = document.getElementById('board');
  boardEl.innerHTML = '';

  for (let vi = 0; vi < 64; vi++) {
    const r = flipped ? 7 - Math.floor(vi/8) : Math.floor(vi/8);
    const c = flipped ? 7 - (vi%8) : (vi%8);
    const sq = r*8+c;

    const div = document.createElement('div');
    div.className = 'sq ' + ((r+c)%2===0 ? 'light' : 'dark');

    if (selected === sq) div.classList.add('selected');
    if (lastMove && (lastMove.from === sq || lastMove.to === sq)) div.classList.add('last-move');

    const isTarget = validTargets.includes(sq);
    const p = state ? state.board[sq] : {piece:0,symbol:'',color:-1};

    if (state && p.color === state.playerColor && state.turn === state.playerColor && !state.gameOver) {
      div.classList.add('clickable');
    }
    if (isTarget) div.classList.add('clickable');

    if (isTarget && p.piece === 0) {
      div.innerHTML += '<div class="dot"></div>';
    }
    if (isTarget && p.piece !== 0) {
      div.innerHTML += '<div class="capture-ring"></div>';
    }

    if (p.piece !== 0) {
      div.innerHTML += '<span class="piece">' + p.symbol + '</span>';
    }

    // Coordinates
    if (c === 0) div.innerHTML += '<span class="coord coord-row">' + (8-r) + '</span>';
    if (r === 7) div.innerHTML += '<span class="coord coord-col">' + COLS[c] + '</span>';

    div.onclick = () => clickSquare(sq);
    boardEl.appendChild(div);
  }
}

function renderCaptured() {
  const top = document.getElementById('captured-top');
  const bottom = document.getElementById('captured-bottom');

  const wHtml = '<span class="label">白方被吃:</span>' + capturedW.map(p=>'<span>'+SYM[p]+'</span>').join('');
  const bHtml = '<span class="label">黑方被吃:</span>' + capturedB.map(p=>'<span>'+SYM[p]+'</span>').join('');

  top.innerHTML = flipped ? wHtml : bHtml;
  bottom.innerHTML = flipped ? bHtml : wHtml;
}

function renderMoveLog() {
  const el = document.getElementById('move-log');
  el.innerHTML = moveLog.map((m,i) =>
    '<span>' + (i%2===0 ? Math.floor(i/2)+1+'. ' : '') + m + '</span>'
  ).join('');
  if (moveLog.length === 0) el.innerHTML = '<span style="opacity:0.4">等待开局...</span>';
  el.scrollTop = el.scrollHeight;
}

// Init
render();
</script>
</body>
</html>"""


# ─── HTTP Handler ───

class GameHandler(SimpleHTTPRequestHandler):
    """Handle API requests and serve the HTML page."""

    def log_message(self, format, *args):
        # Suppress default logging; only log important stuff
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        if path == '/' or path == '/index.html':
            self._send_html(HTML_PAGE)
        elif path == '/api/new':
            self._handle_new_game(params)
        elif path == '/api/state':
            self._handle_state()
        elif path == '/api/move':
            self._handle_move(params)
        elif path == '/api/ai_move':
            self._handle_ai_move()
        elif path == '/api/undo':
            self._handle_undo()
        else:
            self.send_error(404)

    def _send_json(self, data: dict):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str):
        body = html.encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def _handle_new_game(self, params):
        global game_state, player_color
        player_color = int(params.get('color', 0))
        game_state = AntichessGame()
        self._send_json(get_board_json(game_state))

    def _handle_state(self):
        global game_state
        if game_state is None:
            game_state = AntichessGame()
        self._send_json(get_board_json(game_state))

    def _handle_move(self, params):
        from_sq = int(params.get('from', 0))
        to_sq = int(params.get('to', 0))
        promo = int(params.get('promo', 0))
        result = apply_player_move(from_sq, to_sq, promo)
        self._send_json(result)

    def _handle_ai_move(self):
        global game_state
        move = ai_think()
        result = get_board_json(game_state)
        if move:
            result['aiMove'] = {'from': move.from_sq, 'to': move.to_sq}
        self._send_json(result)

    def _handle_undo(self):
        global game_state
        # Simple undo: restart game (full undo would need history stack)
        # For now, just send current state
        # TODO: implement proper undo with history
        self._send_json(get_board_json(game_state))


# ─── Main ───

def main():
    global engine_model, engine_mcts

    parser = argparse.ArgumentParser(description='Play Antichess vs AlphaZero in browser')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--simulations', type=int, default=800,
                        help='MCTS simulations per move')
    parser.add_argument('--port', type=int, default=8080,
                        help='HTTP server port')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load model
    engine_model = AntichessNet(in_channels=18, num_res_blocks=10, channels=128).to(device)

    if args.model and os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            config = checkpoint.get('config', {})
            engine_model = AntichessNet(
                in_channels=config.get('in_channels', 18),
                num_res_blocks=config.get('num_res_blocks', 10),
                channels=config.get('channels', 128),
            ).to(device)
            engine_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            engine_model.load_state_dict(checkpoint)
        print(f"Loaded model: {args.model}")
    else:
        print("WARNING: No model loaded, using random weights")

    engine_model.eval()
    engine_mcts = MCTS(engine_model, num_simulations=args.simulations)

    # Start server
    server = HTTPServer(('0.0.0.0', args.port), GameHandler)
    print(f"\n{'='*50}")
    print(f"  Antichess AlphaZero Web UI")
    print(f"  Open http://localhost:{args.port} in your browser")
    print(f"  Simulations: {args.simulations}")
    print(f"  Press Ctrl+C to quit")
    print(f"{'='*50}\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == '__main__':
    main()
