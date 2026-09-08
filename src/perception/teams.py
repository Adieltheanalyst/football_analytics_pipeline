from __future__ import annotations
import cv2 
import numpy as np
from sklearn.cluster import KMeans
from src.utils.config import Config

REFEREE_TEAM = -1
UNASSIGNED = -1 

def torso_crop(frame: np.ndarray,box:np.ndarray, cfg: Config) -> np.ndarray | None:

    x1,y1,x2,y2 = box
    height= y2-y1
    width = x2-x1
    if height <8 or width <4:
        return None

    top = int(y1 +height * cfg.teams.crop_top)
    bottom = int(y1+height *cfg.teams.crop_bottom)
    margin=int(width * cfg.teams.crop_side_margin)
    left = int(x1 + margin)
    right = int(x2 - margin)

    frame_h , frame_w = frame.shape[:2]
    top, bottom = max(0, top), min(frame_h, bottom)
    left,right=max(0, left), min(frame_w, right)
    if bottom -top < 4 or right-left < 2:
        return None
    return frame[top:bottom, left:right]

def kit_color(crop: np.ndarray) -> np.ndarray:
    """Saturation-weighted mean HSV of a crop"""

    hsv=cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1,3).astype(np.float32)
    hue, sat, val =hsv[:, 0], hsv[:,1], hsv[:, 2]
    weights = sat/ 255.0
    if weights.sum()<1e-3:
        weights = np.ones_like(weights)

    angle = hue * (2 * np.pi/180.0)
    return np.array([
        np.average(np.cos(angle), weights=weights),
        np.average(np.sin(angle), weights=weights),
        np.average(sat, weights=weights) / 255.0,
        np.average(val, weights=weights) / 255.0,
    ],
    dtype=np.float32,
    )

class TeamClassifier:
    """Two-Cluster Kit classifier, fitted once then applied per frame."""

    def __init__(self,cfg: Config):
        self.cfg = cfg
        self._kmeans: KMeans | None = None
        self._samples: list[np.ndarray] = []

    def collect(self,frame: np.ndarray, boxes: np.ndarray) -> None:
        """Accumulate color samples during the fitting pass."""
        for box in boxes:
            crop = torso_crop(frame, box, self.cfg)
            if crop is not None and crop.size > 0:
                self._samples.append(kit_color(crop))
    def fit(self) -> None:
        if len(self._samples) < 4:
            raise RuntimeError(
                f"Only {len(self._samples)} Kit samples collected - not enough to"
                "separated two teams. Check that player detection is working."

            )
        features = np.stack(self._samples)
        self._kmeans= KMeans(n_clusters=2,n_init= 10, random_state=0).fit(features)

    @property
    def fitted(self)-> bool:
        return self._kmeans is not None

    def predict(self,frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Team index (0 or 1) per box; UNASSIGNED where the crop was unusable."""
        if self._kmeans is None:
            raise RuntimeError("TeamClassifier.fit() must be called before predict().")

        if len(boxes) == 0:
            return np.array([],dtype=int)

        teams = np.full(len(boxes), UNASSIGNED, dtype=int)
        usable_idx, features = [], [] 
        for i, box in enumerate(boxes):
            crop=torso_crop(frame,box, self.cfg)
            if crop is not None and crop.size > 0:
                usable_idx.append(i)
                features.append(kit_color(crop))

        if features:
            teams[usable_idx]=self._kmeans.predict(np.stack(features))
        return teams

def assign_goalkeepers(
        gk_pitch_xy: np.ndarray,
        player_pitch_xy: np.ndarray,
        player_teams: np.ndarray,
)-> np.ndarray:
    """This is because keepers wear different kit from the team so color clustering cannot place them"""
    if len(gk_pitch_xy) == 0:
        return np.array([], dtype=int)
    centroids = {}
    for team in (0,1):
        mask= player_teams == team
        if mask.any():
            centroids[team]= np.nanmean(player_pitch_xy[mask], axis=0)

    if len(centroids) <2:
        return np.full(len(gk_pitch_xy), UNASSIGNED,dtype=int)

    assignments = np.full(len(gk_pitch_xy), UNASSIGNED, dtype=int)
    for i, position in enumerate(gk_pitch_xy):
        if np.isnan(position).any():
            continue
        distances = {t: np.linalg.norm(position-c) for t, c in centroids.items()}
        assignments[i] = min(distances, key=distances.get)
    return assignments