from src.utils.config import load_config
from src.perception.detect import Detector
from src.perception.track import PlayerTracker
from src.utils.video import sample_frames

cfg = load_config("config/config.yaml")
detector = Detector(cfg)
tracker=PlayerTracker(cfg)

drops = 0
for idx, ts, frame in sample_frames("data/2e57b9_0.mp4", cfg.video.target_fps):
    raw = detector.detect_players(frame)
    tracked= tracker.update(raw)
    lost=len(raw) - len(tracked)
    if lost>0:
        drops+= lost
    for idx in (74,75,76,77,78,138,139):
        print(f"frame {idx:3d}: detected {len(raw):2d} -> tracked {len(tracked):2d} (lost {lost})")

    if idx > 140:
        break
    print(f"\ntotal detections discarded by tracker: {drops}")