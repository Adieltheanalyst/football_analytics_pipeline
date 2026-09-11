from __future__ import annotations
import cv2
import numpy as np
import supervision as sv
from sports.configs.soccer import SoccerPitchConfiguration
from src.utils.config import Config

PITCH = SoccerPitchConfiguration()
PITCH_VERTICES = np.array(PITCH.vertices, dtype=np.float32)

class PitchCalibrator:
    """Solves and smooths the image pitch homography frame by frame"""

    def __init__(self,cfg: Config):
        self.cfg=cfg
        self.min_keypoints = int(cfg.pitch.min_keypoints)
        self.kp_conf = float(cfg.pitch.keypoint_confidence)
        self.ransac_threshold=float(cfg.pitch.ransac_threshold)
        self.alpha = float(cfg.pitch.smoothing_alpha)

        self._smoothed: np.ndarray | None = None

        self.frames_solved = 0
        self.frames_failed = 0

    def solve(self, keypoints: sv.KeyPoints) -> np.ndarray | None:
        image_pts, pitch_pts=  self._confident_pairs(keypoints)
        if len(image_pts) < self.min_keypoints:
            self.frames_failed += 1
            return self._smoothed

        homography,_ = cv2.findHomography(
            image_pts,pitch_pts,cv2.RANSAC,self.ransac_threshold
        )
        if homography is None:
            self.frames_failed += 1
            return self._smoothed

        self.frames_solved += 1
        self._smoothed = self._smooth(homography)
        return self._smoothed
    def _confident_pairs(
            self, keypoints: sv.KeyPoints
    )-> tuple[np.ndarray,np.ndarray]:
        if keypoints.xy is None or len(keypoints.xy) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)
 
        xy = np.asarray(keypoints.xy[0], dtype=np.float32)        # (32, 2)
        conf_attr = getattr(keypoints, "keypoint_confidence", None)
        if conf_attr is None:
            conf_attr = keypoints.confidence
        conf = (
            np.asarray(conf_attr[0], dtype=np.float32)
            if conf_attr is not None
            else np.ones(len(xy), dtype=np.float32)
        )
 
        mask = conf >= self.kp_conf
        # A keypoint at exactly (0, 0) is the model's way of saying "not found".
        h,w =1080,1920 
        edge=2.0
        mask &= (xy[:, 0] > edge) & (xy[:, 0] < w - edge)
        mask &= (xy[:, 1] > edge) & (xy[:, 1] < h - edge)
        return xy[mask], PITCH_VERTICES[mask]
 
    def _smooth(self, homography: np.ndarray) -> np.ndarray:
        """EMA over matrix elements, normalised so H[2,2] == 1 before blending."""
        h = homography / homography[2, 2]
        if self._smoothed is None:
            return h
        blended = self.alpha * self._smoothed + (1.0 - self.alpha) * h
        return blended / blended[2, 2]
 
    # -- applying -----------------------------------------------------------
 
    @staticmethod
    def to_pitch(points: np.ndarray, homography: np.ndarray | None) -> np.ndarray:
        """
        Map image points (N, 2) to pitch centimetres (N, 2).
 
        Returns NaN rows when there is no homography, so downstream code can
        tell "off the pitch" from "we couldn't calibrate this frame".
        """
        points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if homography is None or len(points) == 0:
            return np.full((len(points), 2), np.nan, dtype=np.float32)
 
        transformed = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2).astype(np.float32), homography
        )
        return transformed.reshape(-1, 2)
 
    # -- reporting ----------------------------------------------------------
 
    @property
    def solve_rate(self) -> float:
        total = self.frames_solved + self.frames_failed
        return self.frames_solved / total if total else 0.0
 
    def reset(self) -> None:
        self._smoothed = None
        self.frames_solved = 0
        self.frames_failed = 0
 
 
def on_pitch_mask(pitch_xy: np.ndarray, margin_cm: float = 500.0) -> np.ndarray:
    """
    Flag points that landed implausibly far outside the pitch.
 
    A bad homography produces coordinates hundreds of metres away. Rather than
    silently keeping them and poisoning possession stats, mark them so the
    event layer can drop them.
    """
    if len(pitch_xy) == 0:
        return np.array([], dtype=bool)
    x, y = pitch_xy[:, 0], pitch_xy[:, 1]
    return (
        (x >= -margin_cm)
        & (x <= PITCH.length + margin_cm)
        & (y >= -margin_cm)
        & (y <= PITCH.width + margin_cm)
    )