"""
AlphaZero-style neural network for Antichess.
Architecture: ResNet with dual heads (policy + value).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from game import ACTION_SIZE


class ResBlock(nn.Module):
    """Residual block with two conv layers + batch norm + skip connection."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class AntichessNet(nn.Module):
    """
    Neural network that takes a board state and outputs:
      - policy: probability distribution over all actions (ACTION_SIZE)
      - value: scalar in [-1, 1] estimating win probability from current player's view

    Architecture follows AlphaZero:
      Input -> Conv -> N x ResBlock -> PolicyHead + ValueHead

    Args:
        in_channels: number of input planes (default 18)
        num_res_blocks: number of residual blocks (default 10)
        channels: number of channels in residual tower (default 128)
    """

    def __init__(self, in_channels: int = 18, num_res_blocks: int = 10, channels: int = 128):
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, channels, 3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(channels)

        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(channels) for _ in range(num_res_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 32, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, ACTION_SIZE)

        # Value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (batch, in_channels, 8, 8) board encoding
        Returns:
            policy_logits: (batch, ACTION_SIZE) raw logits
            value: (batch, 1) in [-1, 1]
        """
        # Shared trunk
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)

        # Policy head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)  # raw logits

        # Value head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

    def predict(self, state: np.ndarray) -> tuple:
        """
        Predict policy and value for a single state.
        Args:
            state: (18, 8, 8) numpy array
        Returns:
            policy: (ACTION_SIZE,) numpy array of probabilities
            value: float in [-1, 1]
        """
        self.eval()
        with torch.no_grad():
            x = torch.FloatTensor(state).unsqueeze(0)
            if next(self.parameters()).is_cuda:
                x = x.cuda()
            logits, v = self(x)
            policy = F.softmax(logits, dim=1).cpu().numpy()[0]
            value = v.cpu().item()
        return policy, value
