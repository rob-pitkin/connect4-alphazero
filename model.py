from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """
    Single AlphaZero Residual Block:
    Conv2d(3x3, pad=1, bias=False) -> BatchNorm2d -> ReLU ->
    Conv2d(3x3, pad=1, bias=False) -> BatchNorm2d -> (+ skip) -> ReLU
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class ConnectFourNet(nn.Module):
    """
    Dual-headed AlphaZero Network for Connect Four (6x7 board, 7 actions).
    Input shape:  (B, 2, 6, 7)
    Outputs:
      - policy_logits: (B, 7) unnormalized raw logits
      - value:         (B, 1) scalar evaluation in [-1, +1]
    """

    def __init__(self, num_res_blocks: int = 3, num_channels: int = 64):
        super().__init__()
        # 1. Convolutional Stem
        self.conv_stem = nn.Conv2d(2, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(num_channels)

        # 2. Shared Residual Tower
        self.res_tower = nn.ModuleList(
            [ResBlock(num_channels) for _ in range(num_res_blocks)]
        )

        # 3. Policy Head (1x1 conv bottleneck -> flatten -> 7 action logits)
        self.policy_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)

        # 4. Value Head (1x1 conv bottleneck -> flatten -> 64 -> 1 -> tanh)
        self.value_conv = nn.Conv2d(num_channels, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 6 * 7, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.bn_stem(self.conv_stem(x)))
        for block in self.res_tower:
            out = block(out)

        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.flatten(start_dim=1)
        policy_logits = self.policy_fc(p)

        # Value Head
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.flatten(start_dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value
