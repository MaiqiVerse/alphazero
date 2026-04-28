"""
AlphaZero training pipeline with multi-GPU support.

Single GPU:
    python train.py --iterations 100

Multi-GPU (e.g. 4 GPUs):
    torchrun --nproc_per_node=4 train.py --iterations 100

Multi-node (e.g. 2 nodes x 4 GPUs):
    # On node 0:
    torchrun --nnodes=2 --node_rank=0 --master_addr=HOST0 --master_port=29500 \
             --nproc_per_node=4 train.py --iterations 100
    # On node 1:
    torchrun --nnodes=2 --node_rank=1 --master_addr=HOST0 --master_port=29500 \
             --nproc_per_node=4 train.py --iterations 100

Architecture:
    Phase 1 - Self-Play:  each GPU runs independent games in parallel
                          (N total games split across world_size workers)
                          All workers gather data to rank 0
    Phase 2 - Training:   DistributedDataParallel across all GPUs
                          Gradients synchronized automatically
    Phase 3 - Arena:      rank 0 only (evaluation is lightweight)
"""

import os
import random
import argparse
import time
from collections import deque
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from game import AntichessGame, ACTION_SIZE
from model import AntichessNet
from mcts import MCTS
from self_play import SelfPlayWorker, GameStats
from logger import TrainLogger, IterationMetrics


# ─── Distributed Helpers ───

def setup_distributed() -> Tuple[int, int, bool]:
    """
    Initialize distributed training if available.

    Returns:
        rank: global rank of this process (0 = master)
        world_size: total number of processes
        is_distributed: whether we're running distributed
    """
    if 'RANK' in os.environ:
        # Launched via torchrun
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, world_size, True
    else:
        return 0, 1, False


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def print_rank0(msg: str, rank: int = 0):
    """Only print on rank 0 to avoid duplicate output."""
    if rank == 0:
        print(msg, flush=True)


def broadcast_model(model: nn.Module, src: int = 0):
    """Broadcast model parameters from src rank to all ranks."""
    if not dist.is_initialized():
        return
    for param in model.parameters():
        dist.broadcast(param.data, src=src)


def gather_examples(local_examples: List[Tuple], rank: int,
                    world_size: int) -> List[Tuple]:
    """
    Gather self-play examples from all ranks to rank 0.

    Each rank sends its local examples; rank 0 receives all.
    Uses pickle-based gather since examples contain numpy arrays.
    """
    if not dist.is_initialized() or world_size == 1:
        return local_examples

    # Gather to rank 0
    gathered = [None] * world_size
    dist.all_gather_object(gathered, local_examples)

    if rank == 0:
        all_examples = []
        for worker_examples in gathered:
            all_examples.extend(worker_examples)
        return all_examples
    else:
        return []


def broadcast_examples(examples: List[Tuple], rank: int) -> List[Tuple]:
    """Broadcast combined examples from rank 0 to all ranks."""
    if not dist.is_initialized():
        return examples
    result = [examples]
    dist.broadcast_object_list(result, src=0)
    return result[0]


# ─── Self-Play (Parallel) ───

def parallel_self_play(model: nn.Module, config: dict,
                       rank: int, world_size: int,
                       device: torch.device) -> Tuple[List, List[GameStats]]:
    """
    Run self-play games distributed across all GPUs.
    Returns (examples, stats_list).
    """
    total_games = config['self_play_games']
    games_per_rank = total_games // world_size
    if rank < total_games % world_size:
        games_per_rank += 1

    print_rank0(
        f"Self-play: {total_games} games across {world_size} GPU(s) "
        f"({games_per_rank} per GPU)", rank
    )

    model.eval()
    worker = SelfPlayWorker(
        model,
        num_simulations=config['num_simulations'],
        temp_threshold=config['temp_threshold'],
        random_opening_moves=config.get('random_opening_moves', 0),
        resign_threshold=config.get('resign_threshold', -0.95),
        resign_consecutive=config.get('resign_consecutive', 5),
    )

    local_examples = []
    local_stats = []
    for i in range(games_per_rank):
        examples, stats = worker.play_game()
        local_examples.extend(examples)
        local_stats.append(stats)
        if (i + 1) % 10 == 0 or (i + 1) == games_per_rank:
            print(f"  [GPU {rank}] {i+1}/{games_per_rank} games, "
                  f"{len(local_examples)} examples", flush=True)

    # Gather all examples and stats to rank 0
    all_examples = gather_examples(local_examples, rank, world_size)
    all_stats = gather_examples(local_stats, rank, world_size)
    return all_examples, all_stats


# ─── Training (DDP) ───
# Training logic is now inside Trainer class to persist optimizer state.


# ─── Arena (Rank 0 Only) ───

def arena_evaluate(new_model: nn.Module, old_state_dict: dict,
                   config: dict, device: torch.device) -> float:
    """
    Evaluate new model vs old model. Runs on rank 0 only.
    Returns win rate of new model.
    """
    new_model.eval()

    old_model = AntichessNet(
        in_channels=config.get('in_channels', 18),
        num_res_blocks=config.get('num_res_blocks', 10),
        channels=config.get('channels', 128),
    ).to(device)
    old_model.load_state_dict(old_state_dict)
    old_model.eval()

    new_mcts = MCTS(new_model, num_simulations=config['arena_simulations'])
    old_mcts = MCTS(old_model, num_simulations=config['arena_simulations'])

    new_wins = 0
    num_games = config['arena_games']

    for i in range(num_games):
        if i % 2 == 0:
            w_mcts, b_mcts = new_mcts, old_mcts
            new_color = 0
        else:
            w_mcts, b_mcts = old_mcts, new_mcts
            new_color = 1

        game = AntichessGame()
        while True:
            over, winner = game.is_game_over()
            if over:
                break
            mcts = w_mcts if game.turn == 0 else b_mcts
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
            game.apply_move(move)

        if winner == new_color:
            new_wins += 1

        if (i + 1) % 10 == 0:
            print(f"  Arena: {i+1}/{num_games}, new wins: {new_wins}", flush=True)

    return new_wins / num_games


# ─── Checkpointing ───

def save_checkpoint(model: nn.Module, iteration: int, config: dict,
                    train_losses: list, path: str = None):
    """Save model checkpoint (rank 0 only)."""
    os.makedirs('checkpoints', exist_ok=True)
    path = path or f'checkpoints/antichess_az_iter{iteration:04d}.pt'
    # If model is DDP-wrapped, save the inner module
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({
        'iteration': iteration,
        'model_state_dict': state_dict,
        'train_losses': train_losses,
        'config': config,
    }, path)
    print(f"Checkpoint saved: {path}", flush=True)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> Tuple[int, list]:
    """Load checkpoint into model."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return checkpoint['iteration'], checkpoint.get('train_losses', [])


# ─── Main Trainer ───

class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.rank, self.world_size, self.is_distributed = setup_distributed()

        if torch.cuda.is_available():
            if self.is_distributed:
                self.device = torch.device(f'cuda:{self.rank % torch.cuda.device_count()}')
            else:
                self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = AntichessNet(
            in_channels=config['in_channels'],
            num_res_blocks=config['num_res_blocks'],
            channels=config['channels'],
        ).to(self.device)

        # Persistent DDP wrapper
        if self.is_distributed:
            self.ddp_model = DDP(self.model, device_ids=[self.device.index],
                                output_device=self.device.index)
        else:
            self.ddp_model = self.model

        # Persistent optimizer — start at near-zero LR, warm-up will ramp it
        self.optimizer = optim.Adam(
            self.ddp_model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
        )

        # LR schedule: linear warm-up → cosine decay
        #
        # Problem without warm-up:
        #   Iteration 1: weights are random, gradients are huge & noisy.
        #   Full LR=2e-3 causes large, erratic parameter updates.
        #   Adam's m_t and v_t are initialized to 0, so early estimates
        #   are biased and unreliable → training can diverge or oscillate.
        #
        # Solution: ramp LR from ~0 to target over `warmup_steps` steps,
        # then cosine decay to eta_min for stable late-stage training.
        warmup_steps = config.get('warmup_steps', 1000)
        self._warmup_steps = warmup_steps

        # Cosine phase scheduler (activated after warm-up)
        # We use a LambdaLR that handles both phases in one function
        lr_min_ratio = 0.01  # eta_min / lr_max

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warm-up: 0 → 1
                return step / max(warmup_steps, 1)
            else:
                # Cosine annealing with warm restarts
                import math
                T_0 = config.get('lr_restart_interval', 10) * config['epochs']
                progress = (step - warmup_steps) % T_0
                return lr_min_ratio + (1 - lr_min_ratio) * 0.5 * (
                    1 + math.cos(math.pi * progress / T_0))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.replay_buffer = deque(maxlen=config['replay_buffer_size'])
        self.iteration = 0
        self.train_losses = []

        # CSV Logger (rank 0 only)
        self.logger = TrainLogger(config.get('log_path', 'logs/train_log.csv')) if self.rank == 0 else None

        param_count = sum(p.numel() for p in self.model.parameters())
        print_rank0(
            f"Initialized on {self.device} | "
            f"World size: {self.world_size} | "
            f"Parameters: {param_count:,}", self.rank
        )

    def _train(self) -> Tuple[float, float, float]:
        """
        Train on replay buffer. Returns (total_loss, policy_loss, value_loss).
        """
        if len(self.replay_buffer) < self.config['min_buffer_size']:
            print_rank0("Not enough data, skipping training.", self.rank)
            return 0.0, 0.0, 0.0

        self.ddp_model.train()

        data = list(self.replay_buffer)
        random.shuffle(data)

        states = np.array([d[0] for d in data], dtype=np.float32)
        pis = np.array([d[1] for d in data], dtype=np.float32)
        values = np.array([d[2] for d in data], dtype=np.float32)

        states_t = torch.FloatTensor(states)
        pis_t = torch.FloatTensor(pis)
        values_t = torch.FloatTensor(values).unsqueeze(1)

        dataset = TensorDataset(states_t, pis_t, values_t)

        if self.is_distributed:
            sampler = DistributedSampler(dataset, num_replicas=self.world_size,
                                         rank=self.rank, shuffle=True)
        else:
            sampler = None

        loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            pin_memory=True,
            num_workers=2,
        )

        total_loss = 0.0
        total_ploss = 0.0
        total_vloss = 0.0
        num_batches = 0

        for epoch in range(self.config['epochs']):
            if sampler is not None:
                sampler.set_epoch(self.iteration * self.config['epochs'] + epoch)

            for batch_states, batch_pis, batch_values in loader:
                batch_states = batch_states.to(self.device, non_blocking=True)
                batch_pis = batch_pis.to(self.device, non_blocking=True)
                batch_values = batch_values.to(self.device, non_blocking=True)

                policy_logits, pred_values = self.ddp_model(batch_states)

                log_probs = torch.log_softmax(policy_logits, dim=1)
                policy_loss = -torch.sum(batch_pis * log_probs, dim=1).mean()
                value_loss = nn.MSELoss()(pred_values, batch_values)
                loss = policy_loss + value_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                total_ploss += policy_loss.item()
                total_vloss += value_loss.item()
                num_batches += 1

        n = max(num_batches, 1)
        avg_loss, avg_ploss, avg_vloss = total_loss / n, total_ploss / n, total_vloss / n

        if self.is_distributed:
            loss_tensor = torch.tensor([avg_loss, avg_ploss, avg_vloss], device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss, avg_ploss, avg_vloss = loss_tensor.tolist()

        return avg_loss, avg_ploss, avg_vloss

    def run(self, num_iterations: int, resume_path: str = None):
        if resume_path and os.path.exists(resume_path):
            self.iteration, self.train_losses = load_checkpoint(
                self.model, resume_path, self.device)
            print_rank0(f"Resumed from {resume_path} (iter {self.iteration})", self.rank)

        broadcast_model(self.model)

        for _ in range(num_iterations):
            self.iteration += 1
            metrics = IterationMetrics(iteration=self.iteration)

            print_rank0(f"\n{'='*60}", self.rank)
            print_rank0(f"  ITERATION {self.iteration}", self.rank)
            print_rank0(f"{'='*60}", self.rank)

            # ── Phase 1: Self-Play ──
            print_rank0("\n[Phase 1] Self-play (parallel)...", self.rank)
            self.model.eval()
            t0 = time.time()

            all_examples, all_stats = parallel_self_play(
                self.model, self.config,
                self.rank, self.world_size, self.device
            )

            sp_time = time.time() - t0
            metrics.self_play_time_s = sp_time

            if self.rank == 0:
                self.replay_buffer.extend(all_examples)
                metrics.replay_buffer_size = len(self.replay_buffer)
                metrics.num_games = len(all_stats)
                if all_stats:
                    metrics.avg_game_length = np.mean([s.num_moves for s in all_stats])
                    resigned = sum(1 for s in all_stats if s.resigned)
                    print(f"Replay buffer: {len(self.replay_buffer)} | "
                          f"Avg len: {metrics.avg_game_length:.0f} | "
                          f"Resigned: {resigned}/{len(all_stats)} | "
                          f"Time: {sp_time:.0f}s", flush=True)

            buffer_list = list(self.replay_buffer) if self.rank == 0 else []
            buffer_list = broadcast_examples(buffer_list, self.rank)
            if self.rank != 0:
                self.replay_buffer = deque(buffer_list, maxlen=self.config['replay_buffer_size'])

            # ── Phase 2: Training ──
            print_rank0("\n[Phase 2] Training (DDP)...", self.rank)
            old_state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
            t0 = time.time()

            total_loss, policy_loss, value_loss = self._train()
            train_time = time.time() - t0

            self.train_losses.append(total_loss)
            metrics.total_loss = total_loss
            metrics.policy_loss = policy_loss
            metrics.value_loss = value_loss
            metrics.learning_rate = self.optimizer.param_groups[0]['lr']
            metrics.train_time_s = train_time

            print_rank0(f"Loss: {total_loss:.4f} (policy={policy_loss:.4f}, "
                        f"value={value_loss:.4f}) | LR: {metrics.learning_rate:.6f} | "
                        f"Time: {train_time:.0f}s", self.rank)

            # ── Phase 3: Arena ──
            accept = True
            if self.rank == 0:
                print("\n[Phase 3] Arena evaluation...", flush=True)
                t0 = time.time()
                win_rate = arena_evaluate(
                    self.model, old_state_dict, self.config, self.device)
                arena_time = time.time() - t0

                metrics.arena_win_rate = win_rate
                metrics.arena_time_s = arena_time
                print(f"Win rate: {win_rate:.1%} | Time: {arena_time:.0f}s", flush=True)

                if win_rate < self.config['win_threshold']:
                    print("Rejected. Reverting.", flush=True)
                    self.model.load_state_dict(old_state_dict)
                    accept = False
                    metrics.model_accepted = False
                else:
                    print("Accepted!", flush=True)
                    metrics.model_accepted = True

                if self.iteration % self.config['save_interval'] == 0:
                    save_checkpoint(self.model, self.iteration,
                                    self.config, self.train_losses)

                # Log metrics
                self.logger.log(metrics)

            if self.is_distributed:
                accept_tensor = torch.tensor([1 if accept else 0], device=self.device)
                dist.broadcast(accept_tensor, src=0)
                broadcast_model(self.model)

        print_rank0("\nTraining complete!", self.rank)
        if self.logger:
            self.logger.close()
            print_rank0(f"Logs saved to: {self.config.get('log_path', 'logs/train_log.csv')}", self.rank)
        cleanup_distributed()


# ─── Configs ───

DEFAULT_CONFIG = {
    'in_channels': 18,
    'num_res_blocks': 10,
    'channels': 128,

    'num_simulations': 800,
    'arena_simulations': 400,

    'self_play_games': 25,         # fewer games, more iterations → faster flywheel
    'temp_threshold': 30,
    'random_opening_moves': 2,     # random first N moves for diversity

    'learning_rate': 2e-3,
    'weight_decay': 1e-4,
    'batch_size': 256,
    'epochs': 5,                   # halved: less overfitting on small batches
    'replay_buffer_size': 200_000, # smaller: old data from weak model expires faster
    'min_buffer_size': 2048,
    'lr_restart_interval': 40,     # cosine restart every 40 iters (was 10 × fewer iters)
    'warmup_steps': 500,           # fewer total steps early on

    'arena_games': 20,             # halved: faster eval, still statistically meaningful
    'win_threshold': 0.55,

    'resign_threshold': -0.95,
    'resign_consecutive': 5,

    'save_interval': 10,
    'log_path': 'logs/train_log.csv',
}

SMALL_CONFIG = {
    'in_channels': 18,
    'num_res_blocks': 5,
    'channels': 64,

    'num_simulations': 100,
    'arena_simulations': 50,

    'self_play_games': 5,          # very fast iteration for testing
    'temp_threshold': 15,
    'random_opening_moves': 2,

    'learning_rate': 2e-3,
    'weight_decay': 1e-4,
    'batch_size': 64,
    'epochs': 3,
    'replay_buffer_size': 20_000,
    'min_buffer_size': 256,
    'lr_restart_interval': 20,
    'warmup_steps': 100,

    'arena_games': 10,
    'win_threshold': 0.55,

    'resign_threshold': -0.95,
    'resign_consecutive': 5,

    'save_interval': 5,
    'log_path': 'logs/train_log_small.csv',
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AlphaZero for Antichess')
    parser.add_argument('--iterations', type=int, default=400,
                        help='Number of training iterations (default 400, was 100 with old config)')
    parser.add_argument('--small', action='store_true')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    config = SMALL_CONFIG if args.small else DEFAULT_CONFIG
    trainer = Trainer(config)
    trainer.run(num_iterations=args.iterations, resume_path=args.resume)