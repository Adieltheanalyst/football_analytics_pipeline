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
    """"""