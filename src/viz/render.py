from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.events.possession import NO_POSSESSION, possession_share
from src.utils.video import count_sampled_frames, sample_frames
from src.viz.radar import COLOUR_BALL, COLOUR_REFEREE, COLOUR_TEAM, Radar

FONT =cv2.FONT_HERSHEY_SIMPLEX

def _colour_for(row) -> tuple[int,int, int]:
    if row.class_name =="referee" or int(row.team) <0:
        return COLOUR_REFEREE
    return COLOUR_TEAM[int(row.team) % len(COLOUR_TEAM)]

def annotate_frame(
        frame: np.ndarray, detections: pd.DataFrame,possessor:int | None
)-> np.ndarray:
    out=frame.copy()

    for row in detections.itertuples(index=False):
        x1,y1,x2,y2= int(row.x1), int(row.y1), int(row.x2), int(row.y2)
        if row.class_name== "ball":
            center = ((x1+x2)//2, (y1+y2)//2)
            cv2.circle(out,center,12,COLOUR_BALL,2)
            cv2.drawMarker(out,center,COLOUR_BALL, cv2.MARKER_CROSS, 10, 1)
            continue
        colour = _colour_for(row)
        has_ball=possessor is not None and int(row.track_id) == possessor
        thickness = 3 if has_ball else 2

        cv2.rectangle(out,(x1,y1), (x2,y2),colour,thickness)
        label = f"#{int(row.track_id)}"
        if has_ball:
            label+= " Ball"
        (tw,th), _ = cv2.getTextSize(label,FONT, 0.45,1)
        cv2.rectangle(out, (x1,y1- th -6), (x1 +tw+6, y1), colour, -1)
        cv2.putText(out, label,(x1+3,y1-4),FONT,0.45,
                    (255,255,255),1, cv2.LINE_AA)

    return out

def _overlay_radar(frame: np.ndarray, radar: np.ndarray, margin:int=20)-> np.ndarray:
    """Drop the radar into the bottom- Left conner , semi-transparent"""
    h,w = radar.shape[:2]
    fh,fw = frame.shape[:2]

    scale = min(1.0, (fw*0.32)/w)
    if scale<1.0:
        adar = cv2.resize(radar,(int(w*scale), int(h*scale)))
        h,w = radar.shape[:2]

    y1, y2 =fh-h -margin,fh-margin
    x1,x2= margin,margin +w
    if y1<0 or x2>fw:
        return frame

    region = frame[y1:y2, x1:x2]
    cv2.addWeighted(radar,0.85,region, 0.15,0,region)
    frame[y1:y2, x1:x2] = radar
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2, y2), (220, 220, 220), 2)
    return frame

def render(
        video_path:str | Path,
        detections: pd.DataFrame,
        possession: pd.DataFrame,
        cfg,
        output_path:str| Path = "outputs/annotated.mp4",
)-> Path:
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    by_frame = dict(tuple(detections.groupby("frame_idx")))
    possession_by_frame=possession.set_index("frame_idx")
    shares = possession_share(possession)

    radar=Radar()
    writer = None
    total =count_sampled_frames(video_path,cfg.video.target_fps)

    for frame_idx,timestamp,frame in tqdm(
        sample_frames(video_path,cfg.video.target_fps), total=total, unit="frame"):
        rows = by_frame.get(frame_idx)
        if rows is None:
            continue

        possessor = None
        if frame_idx in possession_by_frame.index:
            track = int(possession_by_frame.loc[frame_idx,"track_id"])
            possessor = track if track != NO_POSSESSION else None
        annotated = annotate_frame(frame,rows, possessor)

        people = rows[rows["class_name"] != "ball"].to_dict("records")
        ball_rows = rows[rows["class_name"]=="ball"]
        ball_xy=(
            (ball_rows.iloc[0]["pitch_x"], ball_rows.iloc[0]["pitch_y"])
            if len(ball_rows)
            else None
        )
        panel =radar.render(people, ball_xy,possessor)
        panel=radar.with_scoreboard(panel,shares,timestamp)
        annotated = _overlay_radar(annotated,panel)

        if writer is None:
            h,w = annotated.shape[:2]
            writer =cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                cfg.video.target_fps,
                (w,h),
            )
        writer.write(annotated)

    if writer is not None:
        writer.release()
    return output_path