#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from ml_experiment_helpers import (
    ensure_output_dir,
    print_metrics_block,
    print_torch_model_summary,
    save_confusion_matrix_plot,
    save_hyperparam_results,
    save_training_history_plots,
    torch_predict_classes,
)

BATCH_SIZE = 16
EPOCHS_FINAL = 15
EPOCHS_TUNE = 5
IMG_SIZE = 128
NUM_CLASSES = 3
VAL_FRACTION = 0.2
SEED = 42
TRAIN_ROOT = "dataset/train"
TEST_ROOT = "dataset/test"

HYPERPARAM_GRID = [
    {"lr": 1e-3, "dropout": 0.5},
    {"lr": 1e-3, "dropout": 0.3},
    {"lr": 3e-4, "dropout": 0.5},
    {"lr": 1e-2, "dropout": 0.5},
]


class SpectrogramCNN(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)
    return total_loss / max(n, 1), 100.0 * correct / max(n, 1)


@torch.no_grad()
def eval_loss_acc(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        n += labels.size(0)
    return total_loss / max(n, 1), 100.0 * correct / max(n, 1)


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
) -> dict[str, list[float]]:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }
    for _ in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = eval_loss_acc(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)
    return history


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ensure_output_dir("cnn")

    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    full_train = datasets.ImageFolder(root=TRAIN_ROOT, transform=transform)
    test_ds = datasets.ImageFolder(root=TEST_ROOT, transform=transform)
    class_names = full_train.classes
    n_total = len(full_train)
    n_val = max(1, int(n_total * VAL_FRACTION))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_train,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print("Класи:", class_names)
    print(f"Train (підмножина): {n_train}, Val: {n_val}, Test: {len(test_ds)}")

    template = SpectrogramCNN(dropout=0.5).to(device)
    print_torch_model_summary(template, device, (1, IMG_SIZE, IMG_SIZE))

    trials: list[dict] = []
    best: dict | None = None
    for params in HYPERPARAM_GRID:
        set_seed(SEED)
        model = SpectrogramCNN(dropout=params["dropout"]).to(device)
        run_training(model, train_loader, val_loader, device, params["lr"], EPOCHS_TUNE)
        yt, yp = torch_predict_classes(model, val_loader, device)
        vf1 = float(f1_score(yt, yp, average="weighted", zero_division=0))
        trials.append({"params": params, "val_f1_weighted": vf1})
        if best is None or vf1 > best["val_f1_weighted"]:
            best = {"params": params, "val_f1_weighted": vf1}
    assert best is not None
    save_hyperparam_results(trials, best, out / "hyperparameter_search.json")

    full_train_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=True)
    set_seed(SEED)
    final_model = SpectrogramCNN(dropout=best["params"]["dropout"]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(final_model.parameters(), lr=best["params"]["lr"])
    final_history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "test_loss": [],
        "test_acc": [],
    }
    for ep in range(EPOCHS_FINAL):
        tr_loss, tr_acc = train_one_epoch(
            final_model, full_train_loader, criterion, optimizer, device
        )
        va_loss, va_acc = eval_loss_acc(final_model, val_loader, criterion, device)
        te_loss, te_acc = eval_loss_acc(final_model, test_loader, criterion, device)
        final_history["train_loss"].append(tr_loss)
        final_history["val_loss"].append(va_loss)
        final_history["train_acc"].append(tr_acc)
        final_history["val_acc"].append(va_acc)
        final_history["test_loss"].append(te_loss)
        final_history["test_acc"].append(te_acc)
        print(
            f"Epoch [{ep + 1}/{EPOCHS_FINAL}] "
            f"train loss={tr_loss:.4f} acc={tr_acc:.2f}% | "
            f"val loss={va_loss:.4f} acc={va_acc:.2f}% | "
            f"test loss={te_loss:.4f} acc={te_acc:.2f}%"
        )

    save_training_history_plots(final_history, out / "cnn", "CNN")
    torch.save(final_model.state_dict(), out / "cnn_weights.pt")

    full_train_eval_loader = DataLoader(full_train, batch_size=BATCH_SIZE, shuffle=False)
    loaders = [
        (full_train_eval_loader, "Train (повний train)"),
        (val_loader, "Validation (відокремлена підмножина)"),
        (test_loader, "Test"),
    ]
    summary: dict[str, dict] = {}
    for loader, name in loaders:
        yt, yp = torch_predict_classes(final_model, loader, device)
        m = print_metrics_block(name, yt, yp, class_names)
        slug = name.split()[0].lower()
        save_confusion_matrix_plot(
            yt,
            yp,
            class_names,
            out / f"confusion_matrix_{slug}.png",
            f"CNN — {name}",
        )
        summary[name] = {k: v for k, v in m.items() if k != "confusion_matrix"}

    with open(out / "metrics_summary.json", "w", encoding="utf-8") as f:
        json_data = {
            k: {kk: vv for kk, vv in v.items() if kk != "confusion_matrix"} for k, v in summary.items()
        }
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"\nАртефакти збережено в {out.resolve()}")


if __name__ == "__main__":
    main()
