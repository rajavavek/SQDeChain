from pathlib import Path
from sqdechain.data.synthetic import make_synthetic_dataset
from sqdechain.data.loaders import read_table, ensure_text_columns, ensure_image_columns, ensure_behavior_columns


def test_synthetic_data(tmp_path: Path):
    make_synthetic_dataset(tmp_path, n_reviews=20, n_images=5, n_users=10, sequence_length=4, seed=1)
    assert ensure_text_columns(read_table(tmp_path / "text_reviews.csv")).shape[0] == 20
    assert ensure_image_columns(read_table(tmp_path / "image_manifest.csv")).shape[0] == 5
    assert ensure_behavior_columns(read_table(tmp_path / "behavior_logs.csv")).shape[0] == 40
