 
from __future__ import annotations
 
from pathlib import Path
 
import numpy as np
import pandas as pd
import supervision as sv
from tqdm import tqdm
 
from src.perception import cache
from src.perception.detect import Detector
from src.perception.pitch import PitchCalibrator, on_pitch_mask
from src.perception.teams import UNASSIGNED, TeamClassifier, assign_goalkeepers
from src.perception.track import PlayerTracker, anchor_points
from src.utils.config import Config
from src.utils.video import count_sampled_frames, sample_frames
 
 
def _match_ids(raw: sv.Detections, tracked: sv.Detections) -> np.ndarray:
    """
    Map tracker IDs back onto the raw detections; -1 where none matched.
 
    ByteTrack's update_with_detections returns ONLY the detections it could
    associate with an existing track. During fast camera pans that can be a
    small fraction of what was detected — measured at up to 20 of 22 players
    discarded on this clip, 555 detections lost over 150 frames.
 
    Those are real observations with real positions. Dropping them because the
    tracker couldn't name them loses data we already paid for. So we keep the
    full detection set and treat the track ID as an optional enrichment.
    """
    ids = np.full(len(raw), -1, dtype=int)
    if tracked.tracker_id is None or len(tracked) == 0:
        return ids
    lookup = {
        tuple(np.round(box, 2)): int(track_id)
        for box, track_id in zip(tracked.xyxy, tracked.tracker_id)
    }
    for i, box in enumerate(raw.xyxy):
        ids[i] = lookup.get(tuple(np.round(box, 2)), -1)
    return ids
 
 
def _fit_team_classifier(
    video_path: str | Path,
    cfg: Config,
    detector: Detector,
    max_fit_frames: int = 30,
) -> TeamClassifier:
    classifier = TeamClassifier(cfg)
    collected = 0
 
    for _, _, frame in sample_frames(video_path, cfg.video.target_fps):
        players = detector.split_by_class(detector.detect_players(frame))["player"]
        if len(players):
            classifier.collect(frame, players.xyxy)
            collected += 1
        if collected >= max_fit_frames:
            break
 
    classifier.fit()
    return classifier
 
 
def run(video_path: str | Path, cfg: Config, use_cache: bool = True) -> pd.DataFrame:
    """Run the full perception pass, or return the cached table if one exists."""
    if use_cache:
        cached = cache.load(video_path, cfg)
        if cached is not None:
            print(f"Cache hit: {len(cached)} rows (delete the parquet to force a rerun)")
            return cached
 
    detector = Detector(cfg)
    calibrator = PitchCalibrator(cfg)
 
    print("Pass 1/2: fitting team classifier...")
    classifier = _fit_team_classifier(video_path, cfg, detector)
 
    print("Pass 2/2: detection + tracking...")
    detector.reset()
    tracker = PlayerTracker(cfg)
    rows: list[dict] = []
    discarded_by_tracker = 0
 
    total = count_sampled_frames(video_path, cfg.video.target_fps)
    for frame_idx, timestamp, frame in tqdm(
        sample_frames(video_path, cfg.video.target_fps), total=total, unit="frame"
    ):
        # --- people ---------------------------------------------------------
        people = detector.detect_players(frame)
        tracked = tracker.update(people)
        discarded_by_tracker += max(0, len(people) - len(tracked))
        people.tracker_id = _match_ids(people, tracked)
 
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
        # them. Assign by proximity to each team's centroid instead.
        keeper_teams = (
            assign_goalkeepers(
                anchor_points(keepers), anchor_points(players), player_teams
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
                        "pitch_x": float(group_pitch[i][0]),
                        "pitch_y": float(group_pitch[i][1]),
                    }
                )
 
        # --- ball -----------------------------------------------------------
        ball = detector.detect_ball(frame)
        if len(ball):
            x1, y1, x2, y2 = ball.xyxy[0]
            # Bottom-centre, matching the players. A ground-plane homography
            # assumes points lie flat on the pitch; using the box centre
            # projects an airborne ball metres downfield of where it is.
            ball_pitch = PitchCalibrator.to_pitch(
                np.array([[(x1 + x2) / 2, y2]]), homography
            )
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
 
    # A cache write failure should never destroy a completed inference pass.
    try:
        path = cache.save(df, video_path, cfg)
        print(f"Wrote {len(df)} rows to {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Cache write failed (continuing): {exc}")
 
    print(f"Homography solved on {calibrator.solve_rate:.1%} of frames")
    print(f"Detections the tracker could not associate: {discarded_by_tracker} "
          f"(recovered, not dropped)")
    return df
 
 
def summarise(df: pd.DataFrame) -> None:
    """Quick sanity read on a finished perception pass."""
    frames = df["frame_idx"].nunique()
    ball_frames = df[df["class_name"] == "ball"]["frame_idx"].nunique()
    players = df[df["class_name"] == "player"]
    identified = (players["track_id"] >= 0).sum()
 
    print(f"\n  frames processed    {frames}")
    print(f"  ball recall         {ball_frames}/{frames} "
          f"({100 * ball_frames / frames:.1f}%)")
    print(f"  mean players/frame  {len(players) / frames:.1f}")
    print(f"  unique track ids    {players[players.track_id >= 0]['track_id'].nunique()}")
    print(f"  players with an id  {identified}/{len(players)} "
          f"({100 * identified / max(len(players), 1):.1f}%)")
    print(f"  usable pitch coords {df['pitch_x'].notna().sum()}/{len(df)}")
    print("\n  team split:")
    print(players["team"].value_counts().to_string())
 