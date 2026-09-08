



from __future__ import annotations
import hashlib
from pathlib import Path
import pandas as pd
from src.utils.config import Config, config_fingerprint

PERCEPTION_SECTIONS= ("video", "models", "detection", "tracking", "teams","pitch")

DETECTION_COLUMNS = [
    "frame_idx",
    "timestamp",
    "track_id",
    "class_name",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
    "team",
    "pitch_x",
    "pitch_y",
]

def _video_fingerprint(video_path: str | Path)-> str:
    """Identify a video without hashing gigabytes of it."""

    path=Path(video_path)
    digest= hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    size=path.stat().st_size
    digest.update(str(size).encode("utf-8"))

    chunk = 1024 * 1024
    with path.open("rb") as handle:
        digest.update(handle.read(chunk))
        if size > chunk:
            handle.seek(max(0,size-chunk))
            digest.update(handle.read(chunk))
    return digest.hexdigest()[:12]

def cache_path(video_path:str | Path, cfg: Config) -> Path:
    directory = Path(cfg.cache.directory)
    directory.mkdir(parents=True,exist_ok=True)
    key= f"{_video_fingerprint(video_path)}_{config_fingerprint(cfg, PERCEPTION_SECTIONS)}"
    return directory / f"detections_{key}.parquet"

def load(video_path: str | Path , cfg: Config) -> pd.DataFrame | None:
    """Return the cached detections table, or None on a miss"""
    if not cfg.cache.enabled:
        return None
    path=cache_path(video_path, cfg)
    if not path.exists():
        return None
    return pd.read_parquet(path)

def save(df: pd.DataFrame, video_path: str | Path, cfg: Config) -> Path:
    path = cache_path(video_path,cfg)
    missing = set(DETECTION_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Detections table is missing columns: {sorted(missing)}")

    df[DETECTION_COLUMNS].to_parquet(path,index=False)
    return path
