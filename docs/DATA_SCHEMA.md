# SQDeChain Data Schema

The code expects three aligned tourism data sources. Column names can be changed in `prepare_data.py`, but the following schema is recommended for reproducible experiments.

## 1. Text reviews: `text_reviews.csv`

| Column | Type | Required | Description |
|---|---:|---:|---|
| `review_id` | string/int | yes | Unique review identifier. |
| `user_id` | string/int | yes | Anonymous visitor identifier. |
| `entity_id` | string/int | yes | Hotel, restaurant, destination, or provider identifier. |
| `review_text` | string | yes | Review body for DistilDeBERTa-compatible model. |
| `rating` | float/int | yes | Rating on a 1–5 scale or sentiment label. |
| `credibility` | float | no | Reviewer credibility in `[0,1]`; default `1.0`. |
| `recency_weight` | float | no | Recency/trust weight; default `1.0`. |
| `timestamp` | datetime | no | Review creation date. |

Derived fields:

- `q_i`: normalized quality score in `[0,1]`.
- `w_i`: adaptive weight based on recency, credibility, and platform trust.

## 2. Image manifest: `image_manifest.csv`

| Column | Type | Required | Description |
|---|---:|---:|---|
| `image_id` | string/int | yes | Unique image identifier. |
| `user_id` | string/int | yes | Anonymous visitor identifier. |
| `entity_id` | string/int | yes | Destination/provider identifier. |
| `image_path` | string | yes | Relative or absolute path to image file. |
| `text_context` | string | no | Caption/review associated with the image. |
| `label` | int/float | no | Destination attribute or reliability label. |
| `provenance` | float | no | Image provenance score in `[0,1]`; default `1.0`. |
| `text_polarity` | float | no | Associated text polarity in `[0,1]`; default `0.5`. |
| `user_credibility` | float | no | User credibility in `[0,1]`; default `1.0`. |

Derived fields:

- `visual_similarity`: cosine similarity or reliability score produced by MobileViT-compatible model.
- `DIRS`: destination image reliability score.

## 3. Behaviour logs: `behavior_logs.csv`

| Column | Type | Required | Description |
|---|---:|---:|---|
| `user_id` | string/int | yes | Anonymous visitor identifier. |
| `step` | int | yes | Sequential timestep. |
| `visit_frequency` | float | yes | Normalized frequency signal. |
| `duration` | float | yes | Normalized duration/session-length signal. |
| `spending` | float | no | Normalized spending signal. |
| `engagement` | float | no | Normalized engagement score. |
| `loyalty_label` | int/float | yes | Target label/score for repeat loyalty. |

Derived fields:

- `p_i`: GRU-predicted loyalty propensity.
- `TLP`: tokenised loyalty propensity.

## File-size policy

Raw Zenodo data and checkpoints should not be committed to GitHub. Keep them under `data/raw/` and `models/checkpoints/`, which are ignored by `.gitignore`.
