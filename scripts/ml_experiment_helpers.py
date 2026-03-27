from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def ensure_output_dir(name: str) -> Path:
    d = Path("outputs") / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def print_torch_model_summary(model: nn.Module, device: torch.device, in_shape: Tuple[int, ...]) -> None:
    model = model.to(device)
    batch_shape = (1,) + tuple(in_shape)
    try:
        from torchinfo import summary as torchinfo_summary

        print("\n=== model.summary (torchinfo) ===")
        torchinfo_summary(model, input_size=batch_shape, device=str(device))
    except ImportError:
        print("\n=== Архітектура (repr); для табличного summary: pip install torchinfo ===")
        print(model)
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nПараметри: усього {total:,}, тренованих {trainable:,}")
        dummy = torch.zeros(1, *in_shape, device=device)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        print(f"Тестовий вхід {tuple(dummy.shape)} → вихід {tuple(out.shape)}")


def classification_metrics_dict(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    average_weighted: str = "weighted",
) -> Dict[str, Any]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average=average_weighted, zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average=average_weighted, zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average=average_weighted, zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    return out


def print_metrics_block(
    title: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=list(class_names), zero_division=0))
    m = classification_metrics_dict(y_true, y_pred, class_names)
    print(
        "Зведені метрики:\n"
        f"  Accuracy:           {m['accuracy']:.4f}\n"
        f"  Precision (weighted): {m['precision_weighted']:.4f}\n"
        f"  Recall (weighted):    {m['recall_weighted']:.4f}\n"
        f"  F1 (weighted):       {m['f1_weighted']:.4f}\n"
        f"  Precision (macro):    {m['precision_macro']:.4f}\n"
        f"  Recall (macro):       {m['recall_macro']:.4f}\n"
        f"  F1 (macro):           {m['f1_macro']:.4f}"
    )
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion matrix (числа):")
    print(cm)
    m["confusion_matrix"] = cm.tolist()
    return m


def save_confusion_matrix_plot(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    save_path: Path,
    title: str,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("Справжній клас")
    plt.xlabel("Передбачений клас")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_training_history_plots(
    history: Mapping[str, List[float]],
    out_prefix: Path,
    model_name: str,
) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train loss", marker="o")
    if history.get("val_loss"):
        plt.plot(epochs, history["val_loss"], label="Val loss", marker="s")
    plt.xlabel("Епоха")
    plt.ylabel("Loss")
    plt.title(f"{model_name}: втрати")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train acc %", marker="o")
    if history.get("val_acc"):
        plt.plot(epochs, history["val_acc"], label="Val acc %", marker="s")
    plt.xlabel("Епоха")
    plt.ylabel("Accuracy %")
    plt.title(f"{model_name}: точність")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix.parent / f"{out_prefix.name}_curves_loss_acc.png", dpi=200)
    plt.close()

    if history.get("test_loss") and history.get("test_acc"):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history["train_loss"], label="Train", marker="o")
        plt.plot(epochs, history["test_loss"], label="Test (hold-out)", marker="^")
        plt.xlabel("Епоха")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history["train_acc"], label="Train", marker="o")
        plt.plot(epochs, history["test_acc"], label="Test", marker="^")
        plt.xlabel("Епоха")
        plt.ylabel("Accuracy %")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_prefix.parent / f"{out_prefix.name}_curves_train_test.png", dpi=200)
        plt.close()


def save_hyperparam_results(
    trials: List[Dict[str, Any]],
    best: Dict[str, Any],
    path: Path,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"trials": trials, "best": best}, f, indent=2, ensure_ascii=False)
    print(f"\nЗбережено результати підбору гіперпараметрів: {path}")
    print("\n--- Підбір гіперпараметрів (валідаційна метрика) ---")
    for i, t in enumerate(trials, 1):
        score = t.get("val_f1_weighted", t.get("val_f1_weighted_cv_mean"))
        if score is None:
            score_s = "n/a"
        else:
            score_s = f"{float(score):.4f}"
        print(f"  Спроба {i}: params={t.get('params')} → score={score_s}")
    print(f"\nНайкраща конфігурація: {best.get('params')}")
    bscore = best.get("val_f1_weighted", best.get("val_f1_weighted_cv_mean"))
    if bscore is not None:
        print(f"Найкращий score: {float(bscore):.4f}")


def torch_predict_classes(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            pred = torch.argmax(logits, dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    return np.array(y_true), np.array(y_pred)
