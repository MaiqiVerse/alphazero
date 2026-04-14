"""
CSV logger for tracking training metrics across iterations.

Logs: iteration, total_loss, policy_loss, value_loss, arena_win_rate,
      avg_game_length, learning_rate, replay_buffer_size, model_accepted.

Usage:
    logger = TrainLogger('logs/run_001.csv')
    logger.log(iteration=1, total_loss=2.3, policy_loss=1.8, ...)
    logger.close()

View training curves:
    python logger.py logs/run_001.csv
"""

import os
import csv
import time
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class IterationMetrics:
    iteration: int = 0
    timestamp: float = 0.0
    # Training
    total_loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    learning_rate: float = 0.0
    # Self-play
    num_games: int = 0
    avg_game_length: float = 0.0
    self_play_time_s: float = 0.0
    # Arena
    arena_win_rate: float = 0.0
    model_accepted: bool = True
    # Buffer
    replay_buffer_size: int = 0
    train_time_s: float = 0.0
    arena_time_s: float = 0.0


FIELDNAMES = list(IterationMetrics.__dataclass_fields__.keys())


class TrainLogger:
    """Append-only CSV logger. Creates file with header on first write."""

    def __init__(self, path: str = 'logs/train_log.csv'):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        self._file = None
        self._writer = None

        # Resume: if file exists, don't rewrite header
        file_exists = os.path.exists(path) and os.path.getsize(path) > 0
        self._file = open(path, 'a', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDNAMES)
        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def log(self, metrics: IterationMetrics):
        metrics.timestamp = time.time()
        self._writer.writerow(asdict(metrics))
        self._file.flush()

    def close(self):
        if self._file:
            self._file.close()


# ─── CLI: print summary / plot from CSV ───

def print_summary(path: str):
    """Print a formatted summary of training progress."""
    import csv as csv_mod

    with open(path, 'r') as f:
        reader = csv_mod.DictReader(f)
        rows = list(reader)

    if not rows:
        print("Empty log file.")
        return

    print(f"\n{'Iter':>5} {'Loss':>8} {'P-Loss':>8} {'V-Loss':>8} "
          f"{'Arena%':>7} {'Accept':>7} {'AvgLen':>7} {'LR':>10} {'Buffer':>8}")
    print("─" * 85)

    for r in rows:
        accepted = '✓' if r.get('model_accepted', 'True') == 'True' else '✗'
        print(f"{r['iteration']:>5} "
              f"{float(r['total_loss']):>8.4f} "
              f"{float(r['policy_loss']):>8.4f} "
              f"{float(r['value_loss']):>8.4f} "
              f"{float(r['arena_win_rate'])*100:>6.1f}% "
              f"{accepted:>7} "
              f"{float(r['avg_game_length']):>7.1f} "
              f"{float(r['learning_rate']):>10.6f} "
              f"{r['replay_buffer_size']:>8}")

    # Summary stats
    losses = [float(r['total_loss']) for r in rows if float(r['total_loss']) > 0]
    win_rates = [float(r['arena_win_rate']) for r in rows]
    accepted = sum(1 for r in rows if r.get('model_accepted', 'True') == 'True')

    print(f"\n{'─'*85}")
    print(f"Iterations: {len(rows)} | "
          f"Loss: {losses[-1]:.4f} (min {min(losses):.4f}) | "
          f"Arena: {win_rates[-1]*100:.1f}% (max {max(win_rates)*100:.1f}%) | "
          f"Accepted: {accepted}/{len(rows)}")

    # Try to plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        iters = [int(r['iteration']) for r in rows]

        # Loss
        axes[0, 0].plot(iters, [float(r['total_loss']) for r in rows], 'b-', label='Total')
        axes[0, 0].plot(iters, [float(r['policy_loss']) for r in rows], 'r--', label='Policy')
        axes[0, 0].plot(iters, [float(r['value_loss']) for r in rows], 'g--', label='Value')
        axes[0, 0].set_title('Loss'); axes[0, 0].legend(); axes[0, 0].set_xlabel('Iteration')

        # Arena win rate
        axes[0, 1].plot(iters, [float(r['arena_win_rate'])*100 for r in rows], 'b-o', ms=3)
        axes[0, 1].axhline(55, color='r', linestyle='--', alpha=0.5, label='Threshold')
        axes[0, 1].set_title('Arena Win Rate (%)'); axes[0, 1].legend()

        # Game length
        axes[1, 0].plot(iters, [float(r['avg_game_length']) for r in rows], 'g-')
        axes[1, 0].set_title('Avg Game Length'); axes[1, 0].set_xlabel('Iteration')

        # Learning rate
        axes[1, 1].plot(iters, [float(r['learning_rate']) for r in rows], 'm-')
        axes[1, 1].set_title('Learning Rate'); axes[1, 1].set_xlabel('Iteration')

        plt.tight_layout()
        plot_path = path.replace('.csv', '.png')
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to: {plot_path}")
    except ImportError:
        print("\n(Install matplotlib for training curve plots)")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python logger.py <path_to_csv>")
    else:
        print_summary(sys.argv[1])
