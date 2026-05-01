"""Probe training for the deception-probe reproduction.

Three architectures, all sklearn-compatible (the MLP is wrapped to expose
predict_proba):

  - logreg   : L2 logistic regression on standardized activations  (Apollo baseline)
  - svm_rbf  : RBF-kernel SVM via Nystroem 512-component approximation, then LR
  - mlp      : 2-layer MLP (8192 -> 256 -> 1) with BCE loss

Inputs to the trainers are numpy arrays (X: [N, hidden_dim] float32, y: [N] int).
For per-token training (the Apollo-faithful setup), N = total response tokens
across REPE training samples, with each token's label inherited from its parent
(prompt, response) pair's instruction.

Train/val split MUST be by `pair_id`, not by sample — otherwise the same fact
text leaks across both sides of the split (each fact appears twice in REPE,
once with an honest preamble and once with a deceptive one).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def pair_id_split(
    pair_ids: np.ndarray, val_frac: float = 0.2, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_mask, val_mask) over input axis, splitting by pair_id.

    `pair_ids` is per-sample (length N_samples). To use with per-token data,
    propagate via sample_idx: train_token_mask = train_mask[sample_idx].
    """
    unique = np.array(sorted(set(pair_ids.tolist())))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique)
    n_val = max(1, int(len(unique) * val_frac))
    val_pairs = set(unique[:n_val].tolist())
    val_mask = np.array([p in val_pairs for p in pair_ids], dtype=bool)
    return ~val_mask, val_mask


# ---------- linear probe (Apollo baseline) ----------

def train_logreg(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ---------- non-linear probe via Nystrom approximation ----------

def train_svm_rbf(
    X_train: np.ndarray, y_train: np.ndarray,
    C: float = 1.0, n_components: int = 512, gamma=None, seed: int = 42,
) -> Pipeline:
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("nystroem", Nystroem(
            kernel="rbf", n_components=n_components, gamma=gamma,
            random_state=seed,
        )),
        ("clf", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ---------- 2-layer MLP (wrapped to expose .predict_proba like sklearn) ----------

class MLPProbe:
    """Tiny 2-layer MLP. Exposes .predict_proba for sklearn-style scoring."""

    def __init__(self, d_in: int, hidden: int = 256, epochs: int = 30,
                 lr: float = 1e-3, batch_size: int = 512, seed: int = 42):
        import torch, torch.nn as nn
        torch.manual_seed(seed)
        self._torch = torch
        self._nn = nn
        self.model = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden, 1),
        )
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.mean_ = None
        self.std_ = None

    def fit(self, X, y):
        torch, nn = self._torch, self._nn
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        self.mean_ = X_t.mean(0)
        self.std_ = X_t.std(0).clamp(min=1e-6)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        for _ in range(self.epochs):
            perm = torch.randperm(len(X_t))
            for i in range(0, len(X_t), self.batch_size):
                idx = perm[i:i + self.batch_size]
                logits = self.model((X_t[idx] - self.mean_) / self.std_).squeeze(-1)
                loss = loss_fn(logits, y_t[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        return self

    def predict_proba(self, X):
        torch = self._torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model((X_t - self.mean_) / self.std_).squeeze(-1)
            p1 = torch.sigmoid(logits).numpy()
        # match sklearn's [N, 2] shape: P(class=0), P(class=1)
        return np.stack([1 - p1, p1], axis=1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> MLPProbe:
    return MLPProbe(d_in=X_train.shape[1], **kwargs).fit(X_train, y_train)
