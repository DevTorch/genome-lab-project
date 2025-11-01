from __future__ import annotations
import torch
from torch import nn

class CNN1D(nn.Module):
    def __init__(self, in_ch: int = 4, hidden: int = 64, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, 15, padding=7),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden*2, 7, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden*2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
