from __future__ import annotations
import argparse
import platform
import statistics
import time 
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path("models")

def machine_info() -> None:
    print("="*62)
    print("MACHINE")
    print("="*62)
    print(f" platform: {platform.system()} {platform.release()}")
    print(f" processor: {platform.processor()}")
    try:
        import torch

        print(f" torch : {torch.__version__}")
        print(f" cuda : {torch.cuda.is_available()} (expected False here)")
        print(f" threads : {torch.get_num_threads()}")
    except ImportError:
        print(" torch  : not installed")
    print()

def load_frames(video: str | None, n: int, size:int) -> list[np.ndarray]:

    if video:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {video}")
        frames =[]
        while len(frames) < n:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise SystemExit("Video opened but no frames could be read.")
        print(f"Loaded {len(frames)} real frames from {video}")
        return frames
 
    print(f"No --video given; using {n} synthetic {size}x{size} frames.")
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, (size, size, 3), dtype=np.uint8) for _ in range(n)]
 
 
def time_model(model, frames, imgsz: int, warmup: int = 3) -> dict:
    """Warm up, then time each frame individually so we can see the spread."""
    for f in frames[:warmup]:
        model.predict(f, imgsz=imgsz, verbose=False, device="cpu")
 
    per_frame = []
    for f in frames:
        t0 = time.perf_counter()
        model.predict(f, imgsz=imgsz, verbose=False, device="cpu")
        per_frame.append(time.perf_counter() - t0)
 
    total = sum(per_frame)
    return {
        "fps": len(frames) / total,
        "mean_ms": 1000 * statistics.mean(per_frame),
        "median_ms": 1000 * statistics.median(per_frame),
        "p90_ms": 1000 * sorted(per_frame)[int(0.9 * len(per_frame)) - 1],
        "total_s": total,
    }
 
 
def report(label: str, r: dict) -> None:
    print(
        f"  {label:<22} {r['fps']:6.2f} fps   "
        f"mean {r['mean_ms']:6.1f} ms   "
        f"median {r['median_ms']:6.1f} ms   "
        f"p90 {r['p90_ms']:6.1f} ms"
    )
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None, help="optional real clip to benchmark on")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--weights",
        default="yolo11n.pt",
        help="ultralytics auto-downloads this on first run (~6 MB)",
    )
    args = ap.parse_args()
 
    machine_info()
 
    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("ultralytics not installed. Run: pip install -r requirements.txt")
 
    MODELS_DIR.mkdir(exist_ok=True)
    frames = load_frames(args.video, args.frames, args.imgsz)
 
    print()
    print("=" * 62)
    print(f"BENCHMARK  ({len(frames)} frames @ imgsz={args.imgsz})")
    print("=" * 62)
 
    results = {}
 
    # --- PyTorch CPU baseline -------------------------------------------------
    pt_model = YOLO(args.weights)
    results["pytorch"] = time_model(pt_model, frames, args.imgsz)
    report("PyTorch CPU", results["pytorch"])
 
    # --- OpenVINO -------------------------------------------------------------
    ov_dir = Path(args.weights).with_suffix("").name + "_openvino_model"
    try:
        if not Path(ov_dir).exists():
            print(f"\n  exporting to OpenVINO ({ov_dir})... this takes a minute\n")
            YOLO(args.weights).export(format="openvino", imgsz=args.imgsz)
        ov_model = YOLO(ov_dir, task="detect")
        results["openvino"] = time_model(ov_model, frames, args.imgsz)
        report("OpenVINO CPU", results["openvino"])
    except Exception as exc:  # noqa: BLE001 - benchmark should degrade, not crash
        print(f"  OpenVINO CPU           failed: {exc}")
 
    # --- Verdict --------------------------------------------------------------
    print()
    print("=" * 62)
    print("WHAT THIS MEANS")
    print("=" * 62)
    best = max(results.values(), key=lambda r: r["fps"])
    best_name = max(results, key=lambda k: results[k]["fps"])
 
    if "openvino" in results:
        speedup = results["openvino"]["fps"] / results["pytorch"]["fps"]
        print(f"  OpenVINO speedup vs PyTorch: {speedup:.2f}x")
 
    print(f"  Fastest backend: {best_name} at {best['fps']:.1f} fps for ONE model.")
    print()
    two_model_fps = best["fps"] / 2
    print(f"  The pipeline runs two models per frame (players + pitch),")
    print(f"  so budget roughly {two_model_fps:.1f} fps of throughput.")
    for clip_min in (1, 3):
        n = clip_min * 60 * 5  # sampled at 5 fps
        mins = n / two_model_fps / 60
        print(f"    {clip_min}-min clip @ 5 fps sampling = {n} frames "
              f"-> ~{mins:.1f} min per full pass")
    print()
    print("  Rule of thumb: under 3 fps here means drop to nano weights,")
    print("  imgsz=480, or sample at 3 fps. Above 8 fps means you have room.")
    print("  Either way you pay this cost ONCE — after that, read the cache.")
 
 
if __name__ == "__main__":
    main()