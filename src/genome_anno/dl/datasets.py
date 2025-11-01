from __future__ import annotations
from typing import Tuple
import torch
from torch.utils.data import Dataset

class SeqWindowDataset(Dataset):
    """Простой датасет: уже подготовленные one-hot окна и метки.
    X: (N, 4, L), y: (N,)
    """
    def __init__(self, X: torch.Tensor, y: torch.Tensor):
        assert X.ndim == 3 and X.size(1) == 4, "X must be (N,4,L)"
        self.X, self.y = X, y

    def __len__(self) -> int:
        return self.X.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]
