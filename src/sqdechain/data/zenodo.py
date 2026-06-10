from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
from typing import Dict, List

import requests
from tqdm import tqdm


ZENODO_API = "https://zenodo.org/api/records/{record_id}"


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_record_metadata(record_id: str | int, token: str | None = None, timeout: int = 30) -> Dict:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = requests.get(ZENODO_API.format(record_id=record_id), headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def download_record(record_id: str | int, output: str | Path, token: str | None = None, overwrite: bool = False) -> List[Path]:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta = fetch_record_metadata(record_id, token=token)
    (output / "zenodo_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    downloaded = []
    files = meta.get("files", [])
    if not files:
        raise RuntimeError("No files were listed in the Zenodo metadata. The record may be restricted or unavailable.")
    for item in files:
        key = item.get("key") or item.get("filename") or "downloaded_file"
        links = item.get("links", {})
        url = links.get("self") or links.get("download")
        if not url:
            continue
        target = output / key
        if target.exists() and not overwrite:
            downloaded.append(target)
            continue
        with requests.get(url, stream=True, timeout=60, headers={"Authorization": f"Bearer {token}"} if token else {}) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with target.open("wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=key) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        checksum = item.get("checksum", "")
        if checksum.startswith("md5:"):
            actual = _md5(target)
            expected = checksum.split(":", 1)[1]
            if actual != expected:
                raise RuntimeError(f"Checksum mismatch for {target.name}: expected {expected}, got {actual}")
        downloaded.append(target)
    return downloaded
