from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm
from src.perception import cache
from src.perception.detect import Detector
from src.perception.teams import TeamClassifier, UNASSIGNED, assign_goalkeepers
from src.perception.track import PlayerTracker, anchor_points
from src.utils.config import Config
from src.utils.video import sample_frames, count_sampled_frames
from src.perception.pitch import PitchCalibrator, on_pitch_mask


def _fit_team_classifier(
        video_path:str | Path,
        cfg: Config,
        detector:Detector,
        max_fit_frames:int = 30,
) -> TeamClassifier:
    classifier= TeamClassifier(cfg)
    collected=0

    for _, _,frame in sample_frames(video_path,cfg.video.target_fps):
        players=detector.split_by_class(detector.detect_players(frame))["player"]
        if len(players):
            classifier.collect(frame,players.xyxy)
            collected += 1
        if collected >= max_fit_frames:
            break
    classifier.fit()
    return classifier

def run(video_path: str | Path, cfg: Config, use_cache:bool = True)-> pd.DataFrame:
    """Run the full perception pass or return the cached table if one exists."""
    if use_cache:
        cached = cache.load(video_path,cfg)
        if cached is not None:
            print(f"Cache hit: {len(cached)} rows (delete the parquet to force a rerun)")
            return cached
    tracker = PlayerTracker(cfg)
    calibrator = PitchCalibrator(cfg)
    detector = Detector(cfg)
    print("Pass 1/2: fitting team classifier...")
    classifier= _fit_team_classifier(video_path,cfg, detector)
    print("Pass 2/2: detection +tracking...")
    detector.reset()
    tracker= PlayerTracker(cfg)
    rows: list[dict] = []
    for frame_idx,timestamp, frame in tqdm(sample_frames(video_path,cfg.video.target_fps), unit="frame"):
        # timestamp=frame_idx / cfg.video.target_fps
 
        # --- people ----------------------------------------------------------
        people = detector.detect_players(frame)
        people = tracker.update(people)
        groups = detector.split_by_class(people)
 
        players = groups["player"]
        keepers = groups["goalkeeper"]
        referees = groups["referee"]
 
        player_teams = (
            classifier.predict(frame, players.xyxy)
            if len(players)
            else np.array([], dtype=int)
        )
 
        # Goalkeepers wear a different kit, so colour clustering can't place
        # them. Assign by proximity to each team's centroid instead. Until
        # homography exists we use image-space anchors, which is a rough
        # stand-in — revisit once pitch coordinates are real.
        keeper_teams = (
            assign_goalkeepers(
                anchor_points(keepers),
                anchor_points(players),
                player_teams,
            )
            if len(keepers) and len(players)
            else np.full(len(keepers), UNASSIGNED, dtype=int)
        )
        homography = calibrator.solve(detector.detect_pitch(frame))
        for group, teams in (
            (players, player_teams),
            (keepers, keeper_teams),
            (referees, np.full(len(referees), UNASSIGNED, dtype=int)),
        ):
            group_pitch = PitchCalibrator.to_pitch(anchor_points(group), homography)
            group_pitch[~on_pitch_mask(group_pitch)] = np.nan
            for i in range(len(group)):
                x1, y1, x2, y2 = group.xyxy[i]
                rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp": timestamp,
                        "track_id": int(group.tracker_id[i])
                        if group.tracker_id is not None
                        else -1,
                        "class_name": group.data["class_name"][i]
                        if "class_name" in group.data
                        else "unknown",
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(group.confidence[i])
                        if group.confidence is not None
                        else np.nan,
                        "team": int(teams[i]),
                        "pitch_x": float(group_pitch[i][0]),  # filled once homography lands
                        "pitch_y": float(group_pitch[i][1]),
                    }
                )
 
        # --- ball ------------------------------------------------------------
        ball = detector.detect_ball(frame)
        
        if len(ball):
            x1, y1, x2, y2 = ball.xyxy[0]
            ball_pitch = PitchCalibrator.to_pitch(
                        np.array([[(x1 + x2)/ 2, (y1+y2) /2]]), homography)
            ball_pitch[~on_pitch_mask(ball_pitch)] = np.nan
            rows.append(
                {
                    "frame_idx": frame_idx,
                    "timestamp": timestamp,
                    "track_id": -1,
                    "class_name": "ball",
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(ball.confidence[0]),
                    "team": UNASSIGNED,
                    "pitch_x": float(ball_pitch[0][0]),
                    "pitch_y": float(ball_pitch[0][1]),
                }
            )
 
    df = pd.DataFrame(rows)
    path = cache.save(df, video_path, cfg)
    print(f"Wrote {len(df)} rows to {path}")
    print(f"Homography solved on {calibrator.solve_rate:.1%} of frames")
    return df
 
 
def summarise(df: pd.DataFrame) -> None:
    """Quick sanity read on a finished perception pass."""
    frames = df["frame_idx"].nunique()
    ball_frames = df[df["class_name"] == "ball"]["frame_idx"].nunique()
    players = df[df["class_name"] == "player"]
 
    print(f"\n  frames processed   {frames}")
    print(f"  ball recall        {ball_frames}/{frames} ({100 * ball_frames / frames:.1f}%)")
    print(f"  unique track ids   {players['track_id'].nunique()}")
    print(f"  mean players/frame {len(players) / frames:.1f}   (expect ~20)")
    print(f"  pitch coords       {df['pitch_x'].notna().sum()}/{len(df)} rows")
    print(f"  usable pitch coords {df['pitch_x'].notna().sum()}/{len(df)}")
    print("\n  team split:")
    print(players["team"].value_counts().to_string())