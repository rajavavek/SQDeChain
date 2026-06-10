from __future__ import annotations

from pathlib import Path
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


POSITIVE = ["excellent", "clean", "friendly", "beautiful", "safe", "comfortable", "recommended", "amazing"]
NEGATIVE = ["dirty", "late", "unsafe", "crowded", "bad", "expensive", "rude", "poor"]
NEUTRAL = ["hotel", "tour", "beach", "museum", "restaurant", "family", "booking", "trip"]


def _review_text(rng: np.random.Generator, rating: int) -> str:
    words = []
    sentiment_words = POSITIVE if rating >= 4 else NEGATIVE if rating <= 2 else POSITIVE[:3] + NEGATIVE[:3]
    words.extend(rng.choice(sentiment_words, size=4, replace=True).tolist())
    words.extend(rng.choice(NEUTRAL, size=8, replace=True).tolist())
    rng.shuffle(words)
    return " ".join(words).capitalize() + "."


def make_synthetic_dataset(output: str | Path, n_reviews: int = 500, n_images: int = 80, n_users: int = 200, sequence_length: int = 10, seed: int = 42) -> None:
    output = Path(output)
    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    random.seed(seed)

    # Text reviews
    ratings = np.clip(np.round(rng.normal(3.8, 1.1, size=n_reviews)), 1, 5).astype(int)
    reviews = []
    for i, rating in enumerate(ratings):
        reviews.append({
            "review_id": f"r{i:06d}",
            "user_id": f"u{rng.integers(0, n_users):05d}",
            "entity_id": f"e{rng.integers(0, 30):03d}",
            "review_text": _review_text(rng, int(rating)),
            "rating": int(rating),
            "credibility": float(np.clip(rng.beta(8, 2), 0, 1)),
            "recency_weight": float(np.clip(rng.beta(5, 2), 0, 1)),
            "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(int(rng.integers(0, 180)), unit="D"),
        })
    pd.DataFrame(reviews).to_csv(output / "text_reviews.csv", index=False)

    # Images and manifest
    manifest = []
    for i in range(n_images):
        label = int(rng.integers(0, 2))
        base = int(80 + 120 * label)
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :, 0] = np.clip(base + rng.normal(0, 30, (64, 64)), 0, 255)
        arr[:, :, 1] = np.clip(150 + rng.normal(0, 40, (64, 64)), 0, 255)
        arr[:, :, 2] = np.clip(220 - base / 2 + rng.normal(0, 30, (64, 64)), 0, 255)
        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), f"D{i}", fill=(255, 255, 255))
        filename = f"img_{i:05d}.png"
        img.save(image_dir / filename)
        manifest.append({
            "image_id": f"img{i:05d}",
            "user_id": f"u{rng.integers(0, n_users):05d}",
            "entity_id": f"e{rng.integers(0, 30):03d}",
            "image_path": filename,
            "text_context": "beautiful clean destination" if label else "crowded poor destination",
            "label": label,
            "provenance": float(np.clip(rng.beta(8, 2), 0, 1)),
            "text_polarity": float(0.75 if label else 0.35),
            "user_credibility": float(np.clip(rng.beta(8, 2), 0, 1)),
        })
    pd.DataFrame(manifest).to_csv(output / "image_manifest.csv", index=False)

    # Behaviour logs
    logs = []
    for u in range(n_users):
        base_loyalty = float(np.clip(rng.beta(5, 3), 0, 1))
        for step in range(sequence_length):
            freq = float(np.clip(base_loyalty + rng.normal(0, 0.15), 0, 1))
            dur = float(np.clip(base_loyalty * 0.8 + rng.normal(0, 0.20), 0, 1))
            spend = float(np.clip(base_loyalty * 0.7 + rng.normal(0, 0.25), 0, 1))
            eng = float(np.clip((freq + dur + spend) / 3 + rng.normal(0, 0.05), 0, 1))
            label = int((0.35 * freq + 0.25 * dur + 0.20 * spend + 0.20 * eng) > 0.58)
            logs.append({
                "user_id": f"u{u:05d}",
                "step": step,
                "visit_frequency": freq,
                "duration": dur,
                "spending": spend,
                "engagement": eng,
                "loyalty_label": label,
            })
    pd.DataFrame(logs).to_csv(output / "behavior_logs.csv", index=False)
