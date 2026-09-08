from __future__ import annotations
import numpy as np
import supervision as sv
from src.utils.config import Config

class PlayerTracker:
    """ByteTrack wrapper. Turns per-frame detection into stable track IDs."""

    def __init__(self,cfg: Config):
        self.cfg = cfg
        self.tracker = sv.ByteTrack(
            track_activation_threshold=cfg.tracking.track_activation_threshold,
            lost_track_buffer=cfg.tracking.lost_track_buffer,
            minimum_matching_threshold=cfg.tracking.minimum_matching_threshold,
            frame_rate=cfg.video.target_fps,
        )

    def update(self, detections: sv.Detections) -> sv.Detections:
        """Assign persistent tracker_id values to this frame's detections."""
        return self.tracker.update_with_detections(detections)

    def reset(self) -> None:
        self.tracker.reset()


def anchor_points(detections: sv.Detections) -> np.ndarray:
    """
    Bottom-centre of each box — where the player meets the pitch.

    This is the point homography should transform, not the box centre.
    Using the centre puts players roughly chest-height above the grass and
    skews every pitch coordinate downfield.
    """
    return detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)