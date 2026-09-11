from __future__ import annotations
 
import numpy as np
import pandas as pd
 
NO_POSSESSION = -999
UNIDENTIFIED_HOLDER = -1  
UNASSIGNED_TEAM = -1
 
 
def _ball_positions(df: pd.DataFrame) -> pd.DataFrame:
    """One ball row per frame, indexed by frame, NaN where the ball was missed."""
    ball = df[df["class_name"] == "ball"]
    ball = ball.drop_duplicates(subset="frame_idx", keep="first")
    return ball.set_index("frame_idx")[["pitch_x", "pitch_y", "timestamp"]]
 
 
def _holders(df: pd.DataFrame) -> pd.DataFrame:
    """Players and goalkeepers — the people who can actually hold the ball."""
    return df[df["class_name"].isin(["player", "goalkeeper"])]
 
 
def compute_possession(
    df: pd.DataFrame,
    max_distance_cm: float = 300.0,
    hysteresis_frames: int = 3,
) -> pd.DataFrame:
    """
    Return one row per frame: who has the ball, which team, how far away.
 
    Columns: frame_idx, timestamp, track_id, team, distance_cm
    track_id is NO_POSSESSION when nobody qualifies.
    """
    balls = _ball_positions(df)
    holders = _holders(df)
 
    raw: list[dict] = []
    for frame_idx, frame_group in holders.groupby("frame_idx", sort=True):
        if frame_idx not in balls.index:
            raw.append({"frame_idx": frame_idx, "track_id": NO_POSSESSION,
                        "team": UNASSIGNED_TEAM, "distance_cm": np.nan})
            continue
 
        ball = balls.loc[frame_idx]
        if np.isnan(ball["pitch_x"]) or np.isnan(ball["pitch_y"]):
            raw.append({"frame_idx": frame_idx, "track_id": NO_POSSESSION,
                        "team": UNASSIGNED_TEAM, "distance_cm": np.nan})
            continue
 
        dx = frame_group["pitch_x"].to_numpy() - ball["pitch_x"]
        dy = frame_group["pitch_y"].to_numpy() - ball["pitch_y"]
        distances = np.sqrt(dx**2 + dy**2)
 
        # Players whose own position failed to localise can't be the holder.
        distances = np.where(np.isnan(distances), np.inf, distances)
 
        nearest = int(np.argmin(distances))
        if distances[nearest] > max_distance_cm:
            raw.append({"frame_idx": frame_idx, "track_id": NO_POSSESSION,
                        "team": UNASSIGNED_TEAM, "distance_cm": float(distances[nearest])})
            continue
 
        row = frame_group.iloc[nearest]
        raw.append(
            {
                "frame_idx": frame_idx,
                "track_id": int(row["track_id"]),
                "team": int(row["team"]),
                "distance_cm": float(distances[nearest]),
            }
        )
 
    possession = pd.DataFrame(raw)
    possession = _apply_hysteresis(possession, hysteresis_frames)
 
    timestamps = df.drop_duplicates("frame_idx").set_index("frame_idx")["timestamp"]
    possession["timestamp"] = possession["frame_idx"].map(timestamps)
    return possession[["frame_idx", "timestamp", "track_id", "team", "distance_cm"]]
 
 
def _apply_hysteresis(possession: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Suppress single-frame possession flips.
 
    A challenge between two players makes the nearest-player label alternate
    frame to frame. Requiring N consecutive frames before committing to a new
    holder turns that noise into one clean transition.
    """
    if n <= 1 or possession.empty:
        return possession
 
    committed = possession["track_id"].to_list()
    output = committed.copy()
    current = committed[0]
    streak_value, streak_len = current, 0
 
    for i, candidate in enumerate(committed):
        if candidate == current:
            streak_value, streak_len = current, 0
            output[i] = current
            continue
 
        if candidate == streak_value:
            streak_len += 1
        else:
            streak_value, streak_len = candidate, 1
 
        if streak_len >= n:
            current = candidate
            streak_len = 0
        output[i] = current
 
    result = possession.copy()
    result["track_id"] = output
    # Team must follow the committed holder, not the raw nearest player.
    holder_team = (
        possession.drop_duplicates("track_id").set_index("track_id")["team"]
    )
    result["team"] = result["track_id"].map(holder_team).fillna(UNASSIGNED_TEAM).astype(int)
    return result
 
 
def possession_share(possession: pd.DataFrame) -> dict[int, float]:
    """Fraction of contested frames held by each team. Excludes dead frames."""
    live = possession[possession["team"] != UNASSIGNED_TEAM]
    if live.empty:
        return {}
    counts = live["team"].value_counts(normalize=True)
    return {int(team): float(share) for team, share in counts.items()}
 
 
def extract_events(possession: pd.DataFrame) -> pd.DataFrame:
    """
    Derive discrete events from possession transitions.
 
    A change of holder within the same team is a PASS. A change to the other
    team is a TURNOVER. Transitions into or out of NO_POSSESSION are recorded
    as LOOSE_BALL, since at this level of the pipeline we can't distinguish a
    long ball in flight from a genuine loss.
    """
    events: list[dict] = []
    previous = None
 
    for row in possession.itertuples(index=False):
        current = (row.track_id, row.team)
        if previous is None:
            previous = current
            continue
        if current == previous:
            continue
 
        prev_track, prev_team = previous
        if row.track_id == NO_POSSESSION or prev_track == NO_POSSESSION:
            kind = "loose_ball"
        elif row.team == prev_team:
            kind = "pass"
        else:
            kind = "turnover"
 
        events.append(
            {
                "frame_idx": int(row.frame_idx),
                "timestamp": float(row.timestamp),
                "type": kind,
                "from_track": int(prev_track),
                "to_track": int(row.track_id),
                "from_team": int(prev_team),
                "to_team": int(row.team),
            }
        )
        previous = current
 
    return pd.DataFrame(
        events,
        columns=["frame_idx", "timestamp", "type", "from_track", "to_track",
                 "from_team", "to_team"],
    )
 
 
def summarise(possession: pd.DataFrame, events: pd.DataFrame) -> None:
    total = len(possession)
    live = (possession["team"] != UNASSIGNED_TEAM).sum()
 
    print(f"\n  frames             {total}")
    print(f"  in possession      {live} ({100 * live / total:.1f}%)")
 
    for team, share in sorted(possession_share(possession).items()):
        print(f"  team {team} possession {share:.1%}")
 
    if events.empty:
        print("  no events detected")
        return
 
    print("\n  events:")
    for kind, count in events["type"].value_counts().items():
        print(f"    {kind:<12} {count}")
 