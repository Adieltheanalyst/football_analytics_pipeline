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
        self.pitch_model = self._load(cfg.models.pitch_detection, task="pose")
        self.ball_model = self._load(cfg.models.ball_detection)

        self._last_ball_xy: tuple[float,float] | None = None
        self._consecutive_misses=0

    def _load(self,weights: str, task: str = "detect") -> YOLO:
        path = Path(weights)
        if not path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {path}\n"
                "Run `python -m src.cli setup` for download instructions."
            )
        return YOLO(str(path), task=task)


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
    def detect_ball(self, frame:np.ndarray)-> sv.Detections:
        """Detect the ball using an ROI crop when we have a recent fix."""
        roi_cfg= self.cfg.detection.ball_roi
        use_roi = (
            roi_cfg.enabled and self._last_ball_xy is not None 
            and self._consecutive_misses < roi_cfg.max_misses
        )

        if use_roi:
            detections = self._detect_ball_in_roi(frame, roi_cfg.window)
            if len(detections) > 0:
                self._update_ball_state(detections)
                return detections

        detections = self._detect_ball_full(frame)
        self._update_ball_state(detections)
        return detections

    def _detect_ball_full(self,frame: np.ndarray)-> sv.Detections:
        result = self.ball_model.predict(
            frame,
            conf=self.cfg.detection.conf_ball,
            iou=self.cfg.detection.iou_nms,
            imgsz=self.cfg.video.inference_size,
            device=self.cfg.models.device,
            verbose=False,
        )[0]
        detections= sv.Detections.from_ultralytics(result)
        return self._best_ball(detections[detections.class_id == self.ball_class_id])

    def _detect_ball_in_roi(self, frame: np.ndarray, window : int) -> sv.Detections:
        height, width = frame.shape[:2]
        cx, cy = self._last_ball_xy
        x1=max(0, int(cx - window))
        y1=max(0, int(cy-window))
        x2 = min(width, int(cx+ window))
        y2= min(height, int(cy+window))
        if x2- x1 <32 or y2-y1 <32:
            return sv.Detections.empty()

        crop = frame[y1:y2 , x1:x2]
        result = self.ball_model.predict(
            crop,
            conf=self.cfg.detection.conf_ball,
            iou=self.cfg.detection.iou_nms,
            imgsz = self.cfg.detection.ball_roi.inference_size,
            device=self.cfg.models.device,
            verbose=False,
        )[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = self._best_ball(detections[detections.class_id == self.ball_class_id])

        if len(detections):
            detections.xyxy= detections.xyxy+np.array([x1,y1,x1,y1], dtype=float)
        return detections

    @staticmethod
    def _best_ball(detections: sv.Detections) -> sv.Detections:
        """Keeping only the highest confidence ball.There is exactly only one ball;"""
        if len(detections) <= 1:
            return detections
        best = int(np.argmax(detections.confidence))
        return detections[best: best + 1]

    def _update_ball_state(self, detections:sv.Detections) -> None:
        if len(detections)>0:
            x1,y1,x2,y2 = detections.xyxy[0]
            self._last_ball_xy=((x1+x2) / 2, (y1+y2) / 2)
            self._consecutive_misses=0
        else:
            self._consecutive_misses +=1

    # Keypoints

    def detect_pitch(self,frame: np.ndarray) -> sv.KeyPoints:
        result=self.pitch_model.predict(
            frame,
            conf=self.cfg.pitch.detection_confidence,
            imgsz=self.cfg.video.inference_size,
            device=self.cfg.models.device,
            verbose=False
        )[0]
        if result.keypoints is None:
            return sv.KeyPoints.empty()
        return sv.KeyPoints.from_ultralytics(result)
    def split_by_class(self, detections: sv.Detections) -> dict[str, sv.Detections]:
        """Separate players, goalkeepers and referees — they need different handling."""
        c = self.classes
        return {
            "player": detections[detections.class_id == int(c.player)],
            "goalkeeper": detections[detections.class_id == int(c.goalkeeper)],
            "referee": detections[detections.class_id == int(c.referee)],
        }

    def reset(self)->None:
        self._last_ball_xy = None
        self._consecutive_misses = 0
        
    