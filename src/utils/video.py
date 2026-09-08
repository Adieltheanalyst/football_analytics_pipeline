
from __future__ import annotations
from dataclasses import Path
from typing import Iterator
import cv2
import numpy as np


@dataclass
class VideoInfo:
    path:Path
    native_fps:float
    total_frames: int
    width: int
    height:int

    @property
    def duration(self) -> float:
        return self.total_frames / self.native_fps if self.native_fps else 0.0

def probe(video_path:str | Path) -> VideoInfo:
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    capture =cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")
    try:
        return VideoInfo(
            path=path,
            native_fps=capture.get(cv2.CAP_PROP_FPS) or 25.0,
            total_frames= int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()

def sample_frames(
        video_path:str | Path,
        target_fps: float,
        start_second: float | None=None,
        end_second: float | None = None,
) -> Iterator[tuple[int, float,np.ndarray]]:

    info = probe(video_path)
    stride = max(1, int(round(info.native_fps/ target_fps)))
    start_native= int((start_second or 0) * info.native_fps)
    end_native=(
        int(end_second * info.native_fps) if end_second is not None else info.total_frames
    )
    capture = cv2.VideoCapture(str(video_path))
    if start_native > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_native)

    try: 
        native_idx = start_native
        sampled_idx=0
        while native_idx< end_native:
            ok, frame = capture.read()
            if not ok:
                break
            if (native_idx -  start_native) % stride ==0:
                yield sampled_idx, native_idx/ info.native_fps, frame
                sampled_idx += 1
            native_idx += 1

    finally:
        capture.release()


def count_sampled_frames(
        video_path: str | Path,
        target_fps: float,
        start_second: float | None=None,
        end_second: float | None= None,

)-> int:
    info=probe(video_path)
    stride = max(1, int(round(info.native_fps / target_fps)))
    start_native = int((start_second or 0) *  info.native_fps)
    end_native = (
        int(end_second*info.native_fps) if end_second is not None else info.total_frames
    )
    return max(0, (end_native- start_native + stride -1) // stride)