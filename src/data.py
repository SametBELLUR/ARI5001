from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pandas as pd


TARGET_COL = 'default payment_next_month'


def load_credit_default_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Loads UCI credit default dataset.

    - Drops ID
    - y = 1 means default
    """
    df = pd.read_csv(csv_path)

    if TARGET_COL not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' in CSV. Found: {list(df.columns)[:10]} ...")

    # Drop ID if present
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    y = df[TARGET_COL].astype(int).to_numpy()
    Xdf = df.drop(columns=[TARGET_COL])

    feature_names = list(Xdf.columns)

    # ensure numeric
    X = Xdf.apply(pd.to_numeric, errors='coerce').fillna(0.0).to_numpy(dtype=float)

    return X, y, feature_names


def train_val_test_split(X: np.ndarray, y: np.ndarray, *, seed: int = 42, train=0.7, val=0.15):
    """Stratified split without sklearn (simple, deterministic)."""
    rng = np.random.default_rng(seed)

    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)

    def split_class(idx):
        n = len(idx)
        n_tr = int(train * n)
        n_va = int(val * n)
        tr = idx[:n_tr]
        va = idx[n_tr:n_tr + n_va]
        te = idx[n_tr + n_va:]
        return tr, va, te

    tr0, va0, te0 = split_class(idx0)
    tr1, va1, te1 = split_class(idx1)

    tr = np.concatenate([tr0, tr1])
    va = np.concatenate([va0, va1])
    te = np.concatenate([te0, te1])

    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)

    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])


def standardize(Xtr: np.ndarray, Xva: np.ndarray, Xte: np.ndarray):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-9
    return (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
