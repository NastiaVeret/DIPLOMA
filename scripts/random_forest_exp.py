#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder

_scripts = Path(__file__).resolve().parent
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

from ml_experiment_helpers import (
    ensure_output_dir,
    print_metrics_block,
    save_confusion_matrix_plot,
    save_hyperparam_results,
)

IMG_SIZE = (128, 128)
TRAIN_ROOT = "dataset/train"
TEST_ROOT = "dataset/test"
VAL_FRACTION = 0.2
SEED = 42
N_ITER_SEARCH = 12


def load_dataset(base_dir: str) -> tuple[np.ndarray, np.ndarray]:
    X: list[np.ndarray] = []
    y: list[str] = []
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Немає каталогу: {base_dir}")
    for label in sorted(os.listdir(base_dir)):
        class_dir = os.path.join(base_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for file in sorted(os.listdir(class_dir)):
            img_path = os.path.join(class_dir, file)
            if not os.path.isfile(img_path):
                continue
            try:
                img = Image.open(img_path).convert("L")
                img = img.resize(IMG_SIZE)
                X.append(np.array(img).flatten())
                y.append(label)
            except OSError:
                continue
    return np.array(X), np.array(y)


def print_rf_architecture(rf: RandomForestClassifier) -> None:
    print("\n=== Random Forest (конфігурація / «архітектура») ===")
    print(rf)
    n_trees = rf.n_estimators
    print(f"\nКількість дерев: {n_trees}, max_features: {rf.max_features}, max_depth: {rf.max_depth}")


def main() -> None:
    np.random.seed(SEED)
    out = ensure_output_dir("random_forest")

    print("Завантаження train / test...")
    X_train_full, y_train_full = load_dataset(TRAIN_ROOT)
    X_test, y_test = load_dataset(TEST_ROOT)

    X_tf, X_val, y_tf, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VAL_FRACTION,
        stratify=y_train_full,
        random_state=SEED,
    )

    encoder = LabelEncoder()
    encoder.fit(y_train_full)
    y_tf_e = encoder.transform(y_tf)
    y_val_e = encoder.transform(y_val)
    y_full_e = encoder.transform(y_train_full)
    y_test_e = encoder.transform(y_test)

    print(f"Train повний: {len(X_train_full)}, train для тюнінгу: {len(X_tf)}, val: {len(X_val)}, test: {len(X_test)}")
    print("Класи:", list(encoder.classes_))

    param_distributions = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [None, 16, 24, 32, 48],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", 0.3, 0.5],
    }
    base_rf = RandomForestClassifier(random_state=SEED, n_jobs=-1)
    search = RandomizedSearchCV(
        base_rf,
        param_distributions,
        n_iter=N_ITER_SEARCH,
        cv=3,
        scoring="f1_weighted",
        random_state=SEED,
        n_jobs=-1,
        refit=True,
    )
    print("\nRandomizedSearchCV для RandomForest (cv=3, scoring=f1_weighted)...")
    search.fit(X_tf, y_tf_e)

    trials = []
    for mean, params in zip(search.cv_results_["mean_test_score"], search.cv_results_["params"]):
        trials.append({"params": params, "val_f1_weighted_cv_mean": float(mean)})
    save_hyperparam_results(
        trials,
        {"params": dict(search.best_params_), "val_f1_weighted_cv_mean": float(search.best_score_)},
        out / "hyperparameter_search.json",
    )

    rf_val = RandomForestClassifier(**search.best_params_, random_state=SEED, n_jobs=-1)
    rf_val.fit(X_tf, y_tf_e)
    y_val_pred = rf_val.predict(X_val)
    val_f1_holdout = float(f1_score(y_val_e, y_val_pred, average="weighted", zero_division=0))
    print(f"\nF1 (weighted) на відокремленому val: {val_f1_holdout:.4f}")
    with open(out / "validation_holdout_f1.json", "w", encoding="utf-8") as f:
        json.dump({"val_f1_weighted_holdout": val_f1_holdout, "best_params": search.best_params_}, f, indent=2)

    rf_final = RandomForestClassifier(**search.best_params_, random_state=SEED, n_jobs=-1)
    rf_final.fit(X_train_full, y_full_e)

    print_rf_architecture(rf_final)

    def eval_xy(X, y_enc, name: str, slug: str, model: RandomForestClassifier) -> dict:
        y_pred = model.predict(X)
        m = print_metrics_block(name, y_enc, y_pred, encoder.classes_)
        save_confusion_matrix_plot(
            y_enc, y_pred, encoder.classes_, out / f"confusion_matrix_{slug}.png", f"Random Forest — {name}"
        )
        return {k: v for k, v in m.items() if k != "confusion_matrix"}

    summary = {
        "Train (повний train)": eval_xy(X_train_full, y_full_e, "Train (повний train)", "train", rf_final),
        "Validation (відокремлена підмножина)": eval_xy(
            X_val, y_val_e, "Validation (відокремлена підмножина)", "validation", rf_final
        ),
        "Test": eval_xy(X_test, y_test_e, "Test", "test", rf_final),
    }

    sizes = np.linspace(0.1, 1.0, 10)
    tr_a, va_a, te_a = [], [], []
    for frac in sizes:
        n = max(1, int(len(X_tf) * frac))
        X_sub = X_tf[:n]
        y_sub = y_tf_e[:n]
        clf = RandomForestClassifier(**search.best_params_, random_state=SEED, n_jobs=-1)
        clf.fit(X_sub, y_sub)
        tr_a.append(float(np.mean(clf.predict(X_sub) == y_sub)))
        va_a.append(float(np.mean(clf.predict(X_val) == y_val_e)))
        te_a.append(float(np.mean(clf.predict(X_test) == y_test_e)))

    plt.figure(figsize=(9, 5))
    plt.plot(sizes * 100, tr_a, marker="o", label="Train (підвибірка train_fit)")
    plt.plot(sizes * 100, va_a, marker="s", label="Validation")
    plt.plot(sizes * 100, te_a, marker="^", label="Test")
    plt.xlabel("Розмір підвибірки train_fit (% від train_fit)")
    plt.ylabel("Accuracy")
    plt.title("Random Forest: accuracy vs розмір навчальної підвибірки")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "learning_curve_data_fraction.png", dpi=200)
    plt.close()

    imp = rf_final.feature_importances_.reshape(IMG_SIZE[0], IMG_SIZE[1])
    plt.figure(figsize=(6, 5))
    plt.imshow(imp, cmap="hot")
    plt.colorbar(label="Importance")
    plt.title("RF: важливість ознак (переформатовано у 2D)")
    plt.tight_layout()
    plt.savefig(out / "feature_importance_heatmap.png", dpi=200)
    plt.close()

    joblib.dump(rf_final, out / "random_forest_model.pkl")
    joblib.dump(encoder, out / "label_encoder.pkl")

    with open(out / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nАртефакти збережено в {out.resolve()}")


if __name__ == "__main__":
    main()
