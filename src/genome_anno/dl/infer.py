from __future__ import annotations
import torch
from genome_anno.dl.models import CNN1D

def predict_proba(x: torch.Tensor) -> torch.Tensor:
    """x: (N,4,L) -> probs: (N,2)"""
    m = CNN1D().eval()
    with torch.no_grad():
        return torch.softmax(m(x), dim=1)
