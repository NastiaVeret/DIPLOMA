#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _configure_ssl_for_torch_downloads() -> None:
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE"):
        return
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass


_configure_ssl_for_torch_downloads()

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from ml_experiment_helpers import (
    ensure_output_dir,
    print_metrics_block,
    print_torch_model_summary,
    save_confusion_matrix_plot,
    save_hyperparam_results,
)

BATCH_SIZE = 16
IMG_SIZE = 224
SEED = 42
TRAIN_ROOT = "dataset/train"
TEST_ROOT = "dataset/test"
VAL_FRACTION = 0.2
LEARNING_CURVE_STEPS = 10


def build_transform():
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_feature_extractor(device: torch.device) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        full = models.resnet18(weights=weights)
    except Exception:
        full = models.resnet18(weights="IMAGENET1K_V1")
    in3 = (3, IMG_SIZE, IMG_SIZE)
    print("\n=== ResNet18 — повна модель (до обрізання класифікатора) ===")
    print_torch_model_summary(full, device, in3)
    feature_net = nn.Sequential(*list(full.children())[:-1])
    feature_net.to(device)
    feature_net.eval()
    for p in feature_net.parameters():
        p.requires_grad = False
    print("\n=== Екстрактор ознак (без fc, AdaptiveAvgPool → 512×1×1) ===")
    print_torch_model_summary(feature_net, device, in3)
    return feature_net


@torch.no_grad()
def extract_features(net: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    feats = []
    for images, _ in loader:
        images = images.to(device)
        x = net(images)
        x = x.view(x.size(0), -1)
        feats.append(x.cpu().numpy())
    return np.vstack(feats) if feats else np.zeros((0, 512))


def eval_split(
    svm: SVC,
    X: np.ndarray,
    y_enc: np.ndarray,
    name: str,
    slug: str,
    class_names: np.ndarray,
    out: Path,
) -> dict:
    y_pred = svm.predict(X)
    m = print_metrics_block(name, y_enc, y_pred, class_names)
    save_confusion_matrix_plot(
        y_enc, y_pred, class_names, out / f"confusion_matrix_{slug}.png", f"ResNet+SVM — {name}"
    )
    return {k: v for k, v in m.items() if k != "confusion_matrix"}


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = ensure_output_dir("resnet")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    transform = build_transform()
    train_dataset = datasets.ImageFolder(root=TRAIN_ROOT, transform=transform)
    test_dataset = datasets.ImageFolder(root=TEST_ROOT, transform=transform)
    class_names = list(train_dataset.classes)

    y_str_all = np.array([train_dataset.classes[lab] for _, lab in train_dataset.samples])
    idx = np.arange(len(train_dataset))
    idx_train, idx_val = train_test_split(
        idx,
        test_size=VAL_FRACTION,
        stratify=y_str_all,
        random_state=SEED,
    )
    train_fit_ds = Subset(train_dataset, idx_train.tolist())
    val_ds = Subset(train_dataset, idx_val.tolist())

    train_fit_loader = DataLoader(train_fit_ds, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    full_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Класи:", class_names)
    print(f"Train (повний): {len(train_dataset)}, train для тюнінгу: {len(train_fit_ds)}, val: {len(val_ds)}, test: {len(test_dataset)}")

    feature_net = load_feature_extractor(device)

    print("\nВитяг ознак (train fit)...")
    X_tf = extract_features(feature_net, train_fit_loader, device)
    print("Витяг ознак (val)...")
    X_val = extract_features(feature_net, val_loader, device)
    print("Витяг ознак (full train)...")
    X_full = extract_features(feature_net, full_train_loader, device)
    print("Витяг ознак (test)...")
    X_test = extract_features(feature_net, test_loader, device)

    y_tf_s = y_str_all[idx_train]
    y_val_s = y_str_all[idx_val]
    y_test_s = np.array([test_dataset.classes[lab] for _, lab in test_dataset.samples])

    encoder = LabelEncoder()
    encoder.fit(y_str_all)
    y_tf_e = encoder.transform(y_tf_s)
    y_val_e = encoder.transform(y_val_s)
    y_full_e = encoder.transform(y_str_all)
    y_test_e = encoder.transform(y_test_s)

    param_grid = {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["linear", "rbf"],
        "gamma": ["scale", "auto"],
    }
    base = SVC(random_state=SEED)
    grid = GridSearchCV(
        base,
        param_grid,
        cv=3,
        scoring="f1_weighted",
        n_jobs=-1,
        refit=True,
    )
    print("\nGridSearchCV для SVC (cv=3, scoring=f1_weighted)...")
    grid.fit(X_tf, y_tf_e)

    trials = []
    for mean, params in zip(grid.cv_results_["mean_test_score"], grid.cv_results_["params"]):
        trials.append({"params": params, "val_f1_weighted_cv_mean": float(mean)})
    best_params = dict(grid.best_params_)
    best_cv = float(grid.best_score_)
    save_hyperparam_results(
        trials,
        {"params": best_params, "val_f1_weighted_cv_mean": best_cv},
        out / "hyperparameter_search.json",
    )

    svm_val = SVC(**best_params, random_state=SEED)
    svm_val.fit(X_tf, y_tf_e)
    y_val_pred = svm_val.predict(X_val)
    val_f1_holdout = float(f1_score(y_val_e, y_val_pred, average="weighted", zero_division=0))
    print(f"\nF1 (weighted) на відокремленому val після fit тільки на train_fit: {val_f1_holdout:.4f}")
    with open(out / "validation_holdout_f1.json", "w", encoding="utf-8") as f:
        json.dump(
            {"val_f1_weighted_holdout": val_f1_holdout, "best_params": best_params},
            f,
            indent=2,
            ensure_ascii=False,
        )

    svm_final = SVC(**best_params, random_state=SEED)
    svm_final.fit(X_full, y_full_e)

    cls = np.asarray(encoder.classes_)
    summary = {
        "Train (повний train)": eval_split(svm_final, X_full, y_full_e, "Train (повний train)", "train", cls, out),
        "Validation (відокремлена підмножина)": eval_split(
            svm_final, X_val, y_val_e, "Validation (відокремлена підмножина)", "validation", cls, out
        ),
        "Test": eval_split(svm_final, X_test, y_test_e, "Test", "test", cls, out),
    }

    sizes = np.linspace(0.1, 1.0, LEARNING_CURVE_STEPS)
    tr_acc, va_acc, te_acc = [], [], []
    for frac in sizes:
        n = max(1, int(len(X_tf) * frac))
        X_sub = X_tf[:n]
        y_sub = y_tf_e[:n]
        clf = SVC(**best_params, random_state=SEED)
        clf.fit(X_sub, y_sub)
        tr_acc.append(float(np.mean(clf.predict(X_sub) == y_sub)))
        va_acc.append(float(np.mean(clf.predict(X_val) == y_val_e)))
        te_acc.append(float(np.mean(clf.predict(X_test) == y_test_e)))

    plt.figure(figsize=(9, 5))
    plt.plot(sizes * 100, tr_acc, marker="o", label="Train (підвибірка train_fit)")
    plt.plot(sizes * 100, va_acc, marker="s", label="Validation")
    plt.plot(sizes * 100, te_acc, marker="^", label="Test")
    plt.xlabel("Розмір підвибірки train_fit (% від train_fit)")
    plt.ylabel("Accuracy")
    plt.title("ResNet18 ознаки + SVM: крива залежності від обсягу навчальних даних")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "learning_curve_data_fraction.png", dpi=200)
    plt.close()

    with open(out / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    torch.save(feature_net.state_dict(), out / "resnet18_feature_extractor_head.pth")
    joblib.dump(svm_final, out / "svm_on_resnet_features.pkl")
    joblib.dump(encoder, out / "label_encoder.pkl")

    print(f"\nАртефакти збережено в {out.resolve()}")


if __name__ == "__main__":
    main()
