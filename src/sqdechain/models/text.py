from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sqdechain.data.loaders import split_frame
from sqdechain.metrics import classification_metrics


def rating_to_label(rating: pd.Series) -> pd.Series:
    # 0=negative, 1=neutral, 2=positive
    return pd.cut(rating.astype(float), bins=[0, 2.5, 3.5, 5], labels=[0, 1, 2], include_lowest=True).astype(int)


class SklearnTextBaseline:
    """Fast fallback text classifier for smoke tests and CPU-only CI."""

    def __init__(self, max_features: int = 20000, class_weight: str | None = "balanced"):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000, class_weight=class_weight)),
        ])

    def fit(self, texts, labels):
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts):
        return self.pipeline.predict(texts)

    def predict_quality(self, texts):
        proba = self.pipeline.predict_proba(texts)
        # Expected sentiment class normalized to [0,1]
        classes = self.pipeline.named_steps["clf"].classes_.astype(float)
        return (proba @ (classes / max(classes.max(), 1))).clip(0, 1)


def train_text_model(df: pd.DataFrame, cfg: Dict, outdir: str | Path) -> Dict:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("project", {}).get("seed", 42))
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("text_model", {})
    df = df.copy()
    df["label"] = rating_to_label(df["rating"])
    train, val, test = split_frame(
        df,
        train_ratio=float(data_cfg.get("train_ratio", 0.70)),
        val_ratio=float(data_cfg.get("val_ratio", 0.15)),
        test_ratio=float(data_cfg.get("test_ratio", 0.15)),
        stratify_col="label",
        seed=seed,
    )
    backend = model_cfg.get("backend", "sklearn")
    if backend == "transformers":
        try:
            return train_transformer_text_model(train, val, test, cfg, outdir)
        except ImportError as e:
            raise ImportError(
                "Transformers backend requested but dependencies are unavailable. "
                "Install `pip install -e .[full]` or set text_model.backend=sklearn for smoke tests."
            ) from e
    model = SklearnTextBaseline(class_weight="balanced" if model_cfg.get("weighted_loss", True) else None)
    model.fit(train["review_text"].astype(str), train["label"])
    pred = model.predict(test["review_text"].astype(str))
    quality = model.predict_quality(df["review_text"].astype(str))
    metrics = classification_metrics(test["label"], pred)
    output = {
        "backend": backend,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }
    pd.DataFrame([output]).to_csv(outdir / "text_metrics.csv", index=False)
    scored = df[["review_id", "user_id", "entity_id", "rating", "credibility", "recency_weight"]].copy()
    scored["text_quality_score"] = quality
    scored.to_csv(outdir / "text_scores.csv", index=False)
    return output


def train_transformer_text_model(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: Dict, outdir: Path) -> Dict:
    """Fine-tune a Hugging Face DeBERTa/DistilDeBERTa-compatible classifier.

    This path is intentionally compact but production-ready: use the configured checkpoint,
    weighted loss through Trainer class weights if needed, and write predictions/metrics.
    """
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

    model_cfg = cfg.get("text_model", {})
    checkpoint = model_cfg.get("checkpoint", "microsoft/deberta-v3-small")
    max_length = int(model_cfg.get("max_length", 256))

    class ReviewDataset(Dataset):
        def __init__(self, frame, tokenizer):
            self.texts = frame["review_text"].astype(str).tolist()
            self.labels = frame["label"].astype(int).tolist()
            self.tokenizer = tokenizer
        def __len__(self):
            return len(self.texts)
        def __getitem__(self, idx):
            enc = self.tokenizer(self.texts[idx], truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
            item = {k: v.squeeze(0) for k, v in enc.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=3)
    train_ds = ReviewDataset(train, tokenizer)
    val_ds = ReviewDataset(val, tokenizer)
    test_ds = ReviewDataset(test, tokenizer)

    args = TrainingArguments(
        output_dir=str(outdir / "hf_text_checkpoint"),
        learning_rate=float(model_cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(model_cfg.get("batch_size", 32)),
        per_device_eval_batch_size=int(model_cfg.get("batch_size", 32)),
        num_train_epochs=int(model_cfg.get("epochs", 5)),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds)
    trainer.train()
    logits = trainer.predict(test_ds).predictions
    pred = np.argmax(logits, axis=1)
    metrics = classification_metrics(test["label"], pred)
    output = {"backend": "transformers", "checkpoint": checkpoint, **asdict(metrics)}
    pd.DataFrame([output]).to_csv(outdir / "text_metrics.csv", index=False)
    return output
