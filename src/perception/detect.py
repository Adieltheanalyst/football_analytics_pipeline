from __future__ import annotations
from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO

from src.utils.config import Config 

class Detector:
    """Wraps two YOLO models the pipeline needs"""

    def __init__(self,cfg: Config):
        self.cfg = cfg
        self.classes = cfg.detection.classes
        self.ball_class_id = int(self.classes.ball)

        self.player_model = self._load(cfg.models.player_detection)
        self.pitch_model = self._load(cfg.models.pitch_detection)

        self._last_ball_xy: tuple[float,float] | None = None
        self._consecutive_misses=0

    def _load(self,weights: str) -> YOLO:
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {path}\n"
                "Run `python -m src.cli setup` for download instructions."
            )
        return YOLO(str(path))


    def detect_players(self,frame:np.ndarray) -> sv.Detections:
        """Detect players , ball not included here it is hanlded by detect_ball with its own threshold"""

        result = self.player_model.predict(
            frame,
            conf=self.cfg.detection.conf_player,
            iou=self.cfg.detection.iou_nms,
            imgsz=self.cfg.video.inference_size,
            device=self.cfg.models.device,
            verbose=False,
        )[0]
        detections= sv.Detections.from_ultralytics(result)
        return detections[detections.class_id != self.ball_class_id]

    # Ball
    