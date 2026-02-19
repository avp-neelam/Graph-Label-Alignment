#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
5-seeded 10-fold CV baseline suite:
  1) CatBoost (tree)
  2) MLP
  3) Linear SVM
  4) RBF SVM
  5) Logistic Regression

Supports:
  - Loading from npy/npz/csv
  - Multi-class classification
  - Standardization (for linear models / SVM / MLP)
  - Stratified K-fold
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC

from sklearn.utils.class_weight import compute_class_weight

# CatBoost is optional: pip install catboost
try:
    from catboost import CatBoostClassifier
    _HAS_CATBOOST = True
except Exception:
    _HAS_CATBOOST = False


# ----------------------------
# Utilities
# ----------------------------
def set_global_seed(seed: int) -> None:
    # scikit-learn uses numpy RNG internally; setting numpy seed is sufficient here.
    np.random.seed(seed)

def load_xy(path: str, x_key: str = "X", y_key: str = "y") -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads (X, y) from:
      - .npz: expects arrays under keys x_key/y_key
      - .npy: X only, y must be provided separately (not supported here)
      - .csv: last column assumed label unless y_key is a column name (simple parser)
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        if x_key not in data or y_key not in data:
            raise ValueError(f".npz must contain keys '{x_key}' and '{y_key}'. Keys: {list(data.keys())}")
        X = np.asarray(data[x_key])
        y = np.asarray(data[y_key])
        return X, y

    if ext == ".npy":
        raise ValueError(".npy loading expects only X; please use .npz with both X and y, or provide your own loader.")

    if ext == ".csv":
        # Minimal CSV loader: assumes numeric features, label in last column by default.
        arr = np.genfromtxt(path, delimiter=",", dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("CSV must be 2D with at least 2 columns (features + label).")
        X = arr[:, :-1]
        y = arr[:, -1].astype(int)
        return X, y

    raise ValueError(f"Unsupported file extension: {ext}. Use .npz or .csv.")

def ensure_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_samples, n_features). Got shape {X.shape}")
    return X

def encode_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict]:
    """
    Ensures labels are integer-coded 0..C-1.
    Returns encoded y and mapping metadata.
    """
    y = np.asarray(y)
    # Allow strings/ints
    uniq = np.unique(y)
    # If already 0..C-1 ints, keep.
    if np.issubdtype(uniq.dtype, np.integer) and np.array_equal(uniq, np.arange(len(uniq))):
        meta = {"label_encoding": "identity", "classes": uniq.tolist()}
        return y.astype(int), meta
    # Otherwise map
    class_to_idx = {c: i for i, c in enumerate(uniq.tolist())}
    y_enc = np.vectorize(class_to_idx.get)(y)
    meta = {"label_encoding": "mapped", "classes": uniq.tolist(), "class_to_idx": class_to_idx}
    return y_enc.astype(int), meta

def class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


# ----------------------------
# Model factory
# ----------------------------
@dataclass
class ModelSpec:
    name: str
    estimator: object
    needs_scaling: bool

def make_models(seed: int, n_classes: int, use_class_weights: bool = True) -> List[ModelSpec]:
    """
    Returns model specs. For scaled models, we will wrap with StandardScaler in a Pipeline.
    """
    models: List[ModelSpec] = []

    # Logistic Regression (multinomial where supported)
    # saga supports multinomial + l2; lbfgs is also fine.
    lr = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        # multi_class="auto",
        n_jobs=None,
        class_weight="balanced" if use_class_weights else None,
        random_state=seed,
    )
    models.append(ModelSpec("LogReg", lr, needs_scaling=True))

    # Linear SVM: LinearSVC (hinge-ish) supports class_weight; good baseline.
    lsvm = LinearSVC(
        C=1.0,
        class_weight="balanced" if use_class_weights else None,
        random_state=seed,
        max_iter=20000,
    )
    models.append(ModelSpec("SVM_linear", lsvm, needs_scaling=True))

    # RBF SVM: probability=False for speed; decision_function used internally; we evaluate via accuracy/F1.
    rbf = SVC(
        kernel="rbf",
        C=1.0,
        gamma="scale",
        class_weight="balanced" if use_class_weights else None,
        random_state=seed,
    )
    models.append(ModelSpec("SVM_rbf", rbf, needs_scaling=True))

    # MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size="auto",
        learning_rate="adaptive",
        learning_rate_init=1e-3,
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.1,
        random_state=seed,
    )
    models.append(ModelSpec("MLP", mlp, needs_scaling=True))

    # CatBoost (tree): handles non-scaled features; good default params.
    if _HAS_CATBOOST:
        cb = CatBoostClassifier(
            loss_function="MultiClass" if n_classes > 2 else "Logloss",
            depth=6,
            learning_rate=0.1,
            iterations=800,
            l2_leaf_reg=3.0,
            random_seed=seed,
            verbose=False,
        )
        models.append(ModelSpec("CatBoost", cb, needs_scaling=False))
    else:
        # Keep placeholder so the script still runs.
        pass

    return models

def wrap_if_needed(spec: ModelSpec) -> object:
    if spec.needs_scaling:
        return Pipeline([("scaler", StandardScaler()), ("model", spec.estimator)])
    return spec.estimator


# ----------------------------
# CV runner
# ----------------------------

@dataclass
class FoldResult:
    acc: float
    bal_acc: float
    macro_f1: float


def eval_one_fold(model, X_tr, y_tr, X_te, y_te) -> FoldResult:
    model.fit(X_tr, y_tr)
    y_hat = model.predict(X_te)
    return FoldResult(
        acc=float(accuracy_score(y_te, y_hat)),
        bal_acc=float(balanced_accuracy_score(y_te, y_hat)),
        macro_f1=float(f1_score(y_te, y_hat, average="macro")),
    )


def run_seed(X: np.ndarray, y: np.ndarray, seed: int, n_splits: int = 10, shuffle: bool = True, use_class_weights: bool = True) -> Dict[str, Dict]:
    set_global_seed(seed)

    classes = np.unique(y)
    n_classes = len(classes)
    models = make_models(seed=seed, n_classes=n_classes, use_class_weights=use_class_weights)

    if not _HAS_CATBOOST:
        print("[warn] catboost not installed; skipping CatBoost. Install with: pip install catboost")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    out: Dict[str, Dict] = {}
    for spec in models:
        model = wrap_if_needed(spec)
        fold_metrics: List[FoldResult] = []

        for tr_idx, te_idx in skf.split(X, y):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            fold_metrics.append(eval_one_fold(model, X_tr, y_tr, X_te, y_te))

        accs = np.array([m.acc for m in fold_metrics], dtype=float)
        bals = np.array([m.bal_acc for m in fold_metrics], dtype=float)
        f1s  = np.array([m.macro_f1 for m in fold_metrics], dtype=float)

        out[spec.name] = {
            "folds": {
                "acc": accs.tolist(),
                "bal_acc": bals.tolist(),
                "macro_f1": f1s.tolist(),
            },
            "summary": {
                "acc_mean": float(accs.mean()),
                "acc_std": float(accs.std(ddof=1)),
                "bal_acc_mean": float(bals.mean()),
                "bal_acc_std": float(bals.std(ddof=1)),
                "macro_f1_mean": float(f1s.mean()),
                "macro_f1_std": float(f1s.std(ddof=1)),
            },
        }

    return out


def aggregate_across_seeds(seed_runs: Dict[int, Dict[str, Dict]]) -> Dict[str, Dict]:
    """
    Aggregates model summaries across seeds (mean of fold-means; std across seeds).
    """
    model_names = sorted({m for sd in seed_runs.values() for m in sd.keys()})

    agg: Dict[str, Dict] = {}
    for name in model_names:
        acc_means = []
        bal_means = []
        f1_means = []
        for seed, res in seed_runs.items():
            if name not in res:
                continue
            s = res[name]["summary"]
            acc_means.append(s["acc_mean"])
            bal_means.append(s["bal_acc_mean"])
            f1_means.append(s["macro_f1_mean"])

        acc_means = np.array(acc_means, dtype=float)
        bal_means = np.array(bal_means, dtype=float)
        f1_means  = np.array(f1_means, dtype=float)

        agg[name] = {
            "acc_mean_over_seeds": float(acc_means.mean()),
            "acc_std_over_seeds": float(acc_means.std(ddof=1)) if len(acc_means) > 1 else 0.0,
            "bal_acc_mean_over_seeds": float(bal_means.mean()),
            "bal_acc_std_over_seeds": float(bal_means.std(ddof=1)) if len(bal_means) > 1 else 0.0,
            "macro_f1_mean_over_seeds": float(f1_means.mean()),
            "macro_f1_std_over_seeds": float(f1_means.std(ddof=1)) if len(f1_means) > 1 else 0.0,
            "n_seeds": int(len(acc_means)),
        }

    return agg


def pretty_print(agg: Dict[str, Dict]) -> None:
    def fmt(mu, sd):
        return f"{mu:.4f} ± {sd:.4f}"
    print("\n=== Aggregated over seeds (mean of 10-fold means) ===")
    for name, d in agg.items():
        print(f"\n[{name}] (n_seeds={d['n_seeds']})")
        print(f"  acc     : {fmt(d['acc_mean_over_seeds'], d['acc_std_over_seeds'])}")
        print(f"  bal_acc : {fmt(d['bal_acc_mean_over_seeds'], d['bal_acc_std_over_seeds'])}")
        print(f"  macro_f1: {fmt(d['macro_f1_mean_over_seeds'], d['macro_f1_std_over_seeds'])}")


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None,
                    help="Path to .npz (keys X,y) or .csv (last column is label).")
    ap.add_argument("--xkey", type=str, default="X", help="Key for X in .npz")
    ap.add_argument("--ykey", type=str, default="y", help="Key for y in .npz")
    ap.add_argument("--out", type=str, default="cv_results.json", help="Output JSON file")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="List of seeds")
    ap.add_argument("--folds", type=int, default=10, help="Number of CV folds")
    ap.add_argument("--no_shuffle", action="store_true", help="Disable shuffle in StratifiedKFold")
    ap.add_argument("--no_class_weights", action="store_true", help="Disable class_weight='balanced' where applicable")

    args = ap.parse_args()

    if args.data is None:
        raise SystemExit("Please provide --data path to .npz (X,y) or .csv.")

    X, y = load_xy(args.data, x_key=args.xkey, y_key=args.ykey)
    X = ensure_2d(X)
    y, y_meta = encode_labels(y)

    seed_runs: Dict[int, Dict[str, Dict]] = {}
    for seed in args.seeds:
        print(f"\n=== Seed {seed} | {args.folds}-fold CV ===")
        seed_runs[seed] = run_seed(
            X=X,
            y=y,
            seed=seed,
            n_splits=args.folds,
            shuffle=(not args.no_shuffle),
            use_class_weights=(not args.no_class_weights),
        )

    agg = aggregate_across_seeds(seed_runs)
    pretty_print(agg)

    payload = {
        "data": {
            "shape_X": list(X.shape),
            "n_classes": int(len(np.unique(y))),
            "label_meta": y_meta,
        },
        "config": {
            "seeds": args.seeds,
            "folds": args.folds,
            "shuffle": (not args.no_shuffle),
            "class_weights": (not args.no_class_weights),
        },
        "per_seed": seed_runs,
        "aggregated": agg,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results to: {args.out}")


if __name__ == "__main__":
    main()
