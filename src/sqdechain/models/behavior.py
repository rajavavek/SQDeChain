from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from sqdechain.metrics import classification_metrics

FEATURES = ["visit_frequency", "duration", "spending", "engagement"]


class GRULoyaltyModel(nn.Module):
    def __init__(self, input_size: int = 4, hidden_units: int = 128, dropout: float = 0.3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_units, batch_first=True, dropout=0.0)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_units, 2)

    def forward(self, x):
        out, h = self.gru(x)
        last = out[:, -1, :]
        return self.head(self.dropout(last))


def build_sequences(df: pd.DataFrame, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    xs, ys, users = [], [], []
    for user_id, group in df.sort_values(["user_id", "step"]).groupby("user_id"):
        values = group[FEATURES].astype(float).to_numpy()
        labels = group["loyalty_label"].astype(int).to_numpy()
        if len(values) < sequence_length:
            pad = np.repeat(values[:1], sequence_length - len(values), axis=0)
            values = np.vstack([pad, values])
            labels = np.concatenate([np.repeat(labels[:1], sequence_length - len(labels)), labels])
        xs.append(values[-sequence_length:])
        ys.append(labels[-1])
        users.append(str(user_id))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), users


def train_behavior_model(df: pd.DataFrame, cfg: Dict, outdir: str | Path) -> Dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("project", {}).get("seed", 42))
    torch.manual_seed(seed)
    model_cfg = cfg.get("behavior_model", {})
    device = cfg.get("project", {}).get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seq_len = int(model_cfg.get("sequence_length", 30))
    X, y, users = build_sequences(df, seq_len)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    train_end = int(0.70 * len(idx))
    test_start = int(0.85 * len(idx))
    train_idx = idx[:train_end]
    test_idx = idx[test_start:]
    train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
    test_ds = TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx]))
    loader = DataLoader(train_ds, batch_size=int(model_cfg.get("batch_size", 64)), shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=int(model_cfg.get("batch_size", 64)))
    model = GRULoyaltyModel(input_size=X.shape[-1], hidden_units=int(model_cfg.get("hidden_units", 128)), dropout=float(model_cfg.get("dropout", 0.3))).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=float(model_cfg.get("learning_rate", 1e-3)))
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(int(model_cfg.get("epochs", 20))):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())
    metrics = classification_metrics(y_true, y_pred)
    output = {"backend": "gru", "accuracy": metrics.accuracy, "precision": metrics.precision, "recall": metrics.recall, "f1": metrics.f1}
    pd.DataFrame([output]).to_csv(outdir / "behavior_metrics.csv", index=False)

    with torch.no_grad():
        logits = model(torch.tensor(X).to(device))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    scored = pd.DataFrame({"user_id": users, "loyalty_propensity": prob})
    scored.to_csv(outdir / "behavior_scores.csv", index=False)
    return output
