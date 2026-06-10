from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
torch.set_num_threads(1)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sqdechain.data.loaders import split_frame
from sqdechain.metrics import classification_metrics


class ImageManifestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: str | Path, input_size: int = 64):
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.input_size = input_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = self.image_root / path
        img = Image.open(path).convert("RGB").resize((self.input_size, self.input_size))
        arr = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.tensor(arr), torch.tensor(int(row.get("label", 0)), dtype=torch.long)


class TinyImageCNN(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def _build_model(backend: str, checkpoint: str, num_classes: int):
    if backend == "timm":
        try:
            import timm
            return timm.create_model(checkpoint, pretrained=True, num_classes=num_classes)
        except ImportError as e:
            raise ImportError("timm backend requested. Install `pip install -e .[full]` or set image_model.backend=tiny_cnn.") from e
    return TinyImageCNN(num_classes=num_classes)


def train_image_model(df: pd.DataFrame, cfg: Dict, outdir: str | Path) -> Dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("project", {}).get("seed", 42))
    torch.manual_seed(seed)
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("image_model", {})
    device = cfg.get("project", {}).get("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    df = df.copy()
    df["label"] = df.get("label", 0).astype(int)
    train, val, test = split_frame(df, data_cfg.get("train_ratio", 0.70), data_cfg.get("val_ratio", 0.15), data_cfg.get("test_ratio", 0.15), stratify_col="label", seed=seed)
    input_size = int(model_cfg.get("input_size", 64))
    batch_size = int(model_cfg.get("batch_size", 16))
    backend = model_cfg.get("backend", "tiny_cnn")
    checkpoint = model_cfg.get("checkpoint", "tiny")
    model = _build_model(backend, checkpoint, int(df["label"].nunique())).to(device)
    train_loader = DataLoader(ImageManifestDataset(train, data_cfg.get("image_root", "data/sample/images"), input_size), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(ImageManifestDataset(test, data_cfg.get("image_root", "data/sample/images"), input_size), batch_size=batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=float(model_cfg.get("learning_rate", 1e-3)))
    loss_fn = nn.CrossEntropyLoss()
    epochs = int(model_cfg.get("epochs", 2))
    model.train()
    for _ in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x.to(device))
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(y.numpy().tolist())
    metrics = classification_metrics(y_true, y_pred)
    output = {"backend": backend, "checkpoint": checkpoint, "accuracy": metrics.accuracy, "precision": metrics.precision, "recall": metrics.recall, "f1": metrics.f1}
    pd.DataFrame([output]).to_csv(outdir / "image_metrics.csv", index=False)

    # Score full manifest as reliability/similarity proxy.
    full_loader = DataLoader(ImageManifestDataset(df, data_cfg.get("image_root", "data/sample/images"), input_size), batch_size=batch_size)
    scores = []
    with torch.no_grad():
        for x, _ in full_loader:
            logits = model(x.to(device))
            prob = torch.softmax(logits, dim=1)
            score = prob[:, -1].cpu().numpy()
            scores.extend(score.tolist())
    scored = df[["image_id", "user_id", "entity_id", "provenance", "text_polarity", "user_credibility"]].copy()
    scored["visual_similarity"] = np.clip(scores, 0, 1)
    scored.to_csv(outdir / "image_scores.csv", index=False)
    return output
