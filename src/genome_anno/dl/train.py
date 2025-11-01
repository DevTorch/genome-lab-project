from __future__ import annotations

from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import MulticlassAccuracy

from hydra import compose, initialize_config_dir

from genome_anno.dl.datasets import SeqWindowDataset
from genome_anno.dl.models import CNN1D
from genome_anno.utils.logging import configure_logging
from genome_anno.utils.seed import set_seed


# =========================
# Synthetic dataset with a motif
# =========================
def make_dummy_dataset(n: int = 2000, Lw: int = 128) -> SeqWindowDataset:
    """
    Создаёт синтетический датасет:
      - класс 1 содержит мотив 'TATA' около центра (±2 позиций),
      - класс 0 гарантированно НЕ содержит точный 'TATA' в центральной зоне.
    Возвращает Dataset с X:(N,4,L), y:(N,)
    """
    import random

    def one_hot_seq(seq: str) -> torch.Tensor:
        m = {"A": 0, "C": 1, "G": 2, "T": 3}
        X = torch.zeros(4, len(seq))
        for i, ch in enumerate(seq):
            X[m[ch], i] = 1.0
        return X

    def random_seq(L: int) -> list[str]:
        return [random.choice("ACGT") for _ in range(L)]

    motif = "TATA"
    mlen = len(motif)
    center = Lw // 2

    X_list, y_list = [], []
    half = n // 2

    # class 0: ensure no exact 'TATA' in central window
    for _ in range(half):
        s = random_seq(Lw)
        low, high = center - 6, center + 6
        for pos in range(max(0, low), min(Lw - mlen, high)):
            if "".join(s[pos : pos + mlen]) == motif:
                # break the motif by mutating one base
                choices = [b for b in "ACGT" if b != s[pos]]
                s[pos] = random.choice(choices)
        X_list.append(one_hot_seq("".join(s)))
        y_list.append(0)

    # class 1: insert motif near center with small jitter
    for _ in range(n - half):
        s = random_seq(Lw)
        jitter = random.randint(-2, 2)
        pos = max(0, min(Lw - mlen, center - mlen // 2 + jitter))
        s[pos : pos + mlen] = list(motif)
        X_list.append(one_hot_seq("".join(s)))
        y_list.append(1)

    X = torch.stack(X_list, dim=0)  # (N, 4, L)
    y = torch.tensor(y_list, dtype=torch.long)
    idx = torch.randperm(n)
    return SeqWindowDataset(X[idx], y[idx])


# =========================
# Lightning Module
# =========================
class LitModule(L.LightningModule):
    def __init__(self, model: nn.Module, lr: float = 1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

        # Метрики
        self.train_acc = MulticlassAccuracy(num_classes=2)
        self.val_acc = MulticlassAccuracy(num_classes=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    # --- train ---
    def training_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.train_acc.update(preds, y)
        self.log("train/loss_step", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log("train/loss_epoch", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/acc_epoch", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self) -> None:
        self.train_acc.reset()

    # --- validation ---
    def validation_step(self, batch, batch_idx: int):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)

        self.val_acc.update(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=False)
        self.log("val/acc", self.val_acc, prog_bar=True, on_epoch=True, sync_dist=False)

    def on_validation_epoch_end(self) -> None:
        self.val_acc.reset()


def main() -> None:
    configure_logging()
    set_seed(42)

    # Hydra configs: src/genome_anno/config
    config_dir = Path(__file__).resolve().parent.parent / "config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="config")
    print(OmegaConf.to_yaml(cfg))

    # Data
    ds = make_dummy_dataset(cfg.data.n_samples, cfg.data.window_size)
    n_train = int(0.9 * len(ds))
    ds_train, ds_val = random_split(ds, [n_train, len(ds) - n_train])

    dl_train = DataLoader(
        ds_train,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model + Lightning
    model = CNN1D(in_ch=4, hidden=cfg.model.hidden, num_classes=cfg.model.num_classes)
    lit = LitModule(model, lr=cfg.train.lr)

    # Callbacks
    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1)
    early = EarlyStopping(monitor="val/loss", mode="min", patience=3)

    # Trainer
    trainer = L.Trainer(
        max_epochs=cfg.train.trainer.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=cfg.train.trainer.precision,
        callbacks=[ckpt, early],
        log_every_n_steps=10,
        num_sanity_val_steps=0,  # убираем лишний варнинг в sanity-check
    )

    # Fit
    trainer.fit(lit, dl_train, dl_val)

if __name__ == "__main__":
    main()
