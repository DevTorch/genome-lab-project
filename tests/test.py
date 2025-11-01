# tests/test.py
from __future__ import annotations

import torch

from genome_anno.dl.train import make_dummy_dataset, LitModule
from genome_anno.dl.models import CNN1D
from genome_anno.utils.seed import set_seed


def test_dummy_dataset_shapes():
    set_seed(123)
    N, L = 64, 128
    ds = make_dummy_dataset(n=N, Lw=L)

    assert len(ds) == N
    x0, y0 = ds[0]
    # X: (4, L), y: scalar {0,1}
    assert isinstance(x0, torch.Tensor) and isinstance(y0, torch.Tensor)
    assert x0.shape == (4, L)
    assert y0.ndim == 0 and y0.item() in (0, 1)


def test_model_forward_shape():
    set_seed(123)
    B, L = 8, 128
    model = CNN1D(in_ch=4, hidden=64, num_classes=2)
    x = torch.randn(B, 4, L)  # (B, C=4, L)
    y = model(x)
    assert y.shape == (B, 2)  # два класса


def test_training_and_validation_steps_run():
    set_seed(123)
    # маленький батч из синтетического датасета
    ds = make_dummy_dataset(n=32, Lw=128)
    X = torch.stack([ds[i][0] for i in range(16)], dim=0).float()  # (16, 4, 128)
    y = torch.tensor([int(ds[i][1]) for i in range(16)], dtype=torch.long)

    lit = LitModule(CNN1D(in_ch=4, hidden=64, num_classes=2), lr=1e-3)

    # один шаг обучения
    loss = lit.training_step((X, y), batch_idx=0)
    assert torch.is_tensor(loss) and loss.item() >= 0.0

    # один шаг валидации (должен пройти без исключений)
    lit.validation_step((X, y), batch_idx=0)
