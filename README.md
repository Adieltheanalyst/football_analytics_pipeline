# football_analytics_pipeline

Turns broadcast football video into structured event data: player and ball
tracking, pitch calibration, and a rule-based layer that resolves possession,
passes and turnovers into a timestamped event stream.

Built and benchmarked on CPU-only hardware. Every number below was measured on
this machine, not copied from a paper.

> **Status:** perception and possession layers are working end to end.
> Visualisation and CLI are in progress — see [Roadmap](#roadmap).

---

## Results

Measured on `2e57b9_0.mp4` (DFL Bundesliga sample, 1920x1080, 30 s),
sampled at 5 fps → 150 frames, 2,710 detections.

| Metric | Result | Notes |
|---|---|---|
| Ball recall | **90.0%** | 135 of 150 frames |
| Homography solve rate | **100%** | ≥6 confident keypoints in every frame |
| Plausible pitch coordinates | **98.2%** | within pitch bounds + 5 m margin |
| Team label stability | **0.95** | mean consistency of a label within one track |
| Frames with possession assigned | **49.3%** | ball within 6 m of a player |
| Inference speed | **0.53 s/frame** | down from 2.18 s/frame — see below |

### Hardware and the OpenVINO result

Everything runs on an **Intel i7-1185G7** (4-core, 15 W ultrabook CPU),
32 GB RAM, **no CUDA GPU**. The three detection models are YOLOv8x variants —
68M parameters, 257 GFLOPs each — and the pipeline runs three of them per frame.

Exporting all three to OpenVINO cut inference from **2.18 s/frame to
0.53 s/frame, a 4.1x speedup**, taking a full pass over the clip from 5.5
minutes to 1.3. Ball recall moved from 92.7% to 90.0% in the process — a small
accuracy cost for a large throughput gain, and the tradeoff is recorded here
rather than buried.

---

## How it works

```
video
  └─ sample at 5 fps
      ├─ player model (YOLOv8x)  ─→ players / goalkeepers / referees
      ├─ ball model   (YOLOv8x)  ─→ ball, with ROI-crop search
      └─ pitch model  (YOLOv8x-pose) ─→ 32 keypoints
                                         └─ RANSAC homography
                                             └─ pitch coordinates (cm)
      ↓
  ByteTrack           ─→ persistent track IDs
  HSV kit clustering  ─→ team labels
      ↓
  Parquet cache  ← expensive, non-deterministic work stops here
      ↓
  possession state machine ─→ possession %, passes, turnovers
```

**The cache boundary is the important design decision.** Inference is slow and
runs once; the event logic is pure pandas over the cached table, runs in
milliseconds, and is unit-testable without a video, a model or a GPU. Tuning a
possession threshold does not mean re-running YOLO.

### Choices worth explaining

**HSV kit clustering instead of SigLIP.** The reference implementation uses
SigLIP embeddings → UMAP → KMeans for team assignment. That is a transformer
forward pass per player crop, which is unworkable on this CPU. Saturation-
weighted mean HSV over the torso region achieves 0.95 label stability at
negligible cost.

**Bottom-centre anchors.** Players are projected from the bottom edge of their
box — where they meet the grass. A ground-plane homography assumes points lie
flat on the pitch; using the box centre projects a player from chest height and
lands them metres downfield. The same applies to the ball.

**Ball ROI search.** Rather than searching the full frame every time, the ball
detector crops a window around the last known position and runs at full input
resolution on that crop. Faster and more accurate, because the ball occupies
more pixels after upscaling.

**Possession threshold from measurement, not intuition.** The nearest-player-
to-ball distance distribution on this clip is median 5.05 m, p25 1.62 m,
p75 8.78 m. The 6 m threshold sits just above the median. Tighter than ~4 m
discards real possession to homography error; looser than ~8 m starts claiming
uncontested balls.

**Hysteresis on possession.** During a challenge the nearest-player label
alternates frame to frame. A new candidate must lead for 3 consecutive frames
before possession switches. This costs a 2-frame commit lag and removes most
spurious pass events.

---

## Limitations

These are measured, not hypothetical. Finding them is a large part of the point.

**Frame sampling breaks tracking association.** 139 unique track IDs were
issued over 150 frames where roughly 25 would be correct. ByteTrack matches
detections between frames by IoU, which assumes consecutive frames. At 5 fps
sampled from 25 fps, players move five times further than expected between
frames, boxes stop overlapping, and identities are re-issued. Raising
`lost_track_buffer` from 30 to 60 improved this from 199 to 139 IDs but does
not solve it. The correct fix is tracking at native frame rate and sampling
only for downstream work — at roughly 5x the inference cost.

**Long passes are not detected as passes.** They register as two `loose_ball`
events instead of one `pass`, because the ball leaves the passer's radius
before entering the receiver's. On this clip: 12 loose_balls, 2 passes,
1 turnover. Fixing this requires interpolating ball trajectory across the gap
rather than treating each frame independently.

**Homography degrades near the horizon.** A perspective transform divides by a
term approaching zero for image points near the vanishing line, so players at
the far edge of the pitch project hundreds of metres away. 1.8% of positions
are affected and are masked to NaN rather than silently corrupting statistics.

**Goalkeeper team assignment is weak.** Keepers wear a different kit, so colour
clustering cannot place them. They are assigned by proximity to each team's
centroid, which is unreliable when the camera shows only part of the pitch.

**Single clip, no ground truth.** All numbers above are internal consistency
checks, not accuracy against hand-labelled data. Possession accuracy in
particular is unvalidated — see Roadmap.

---

## Setup

Requires Python 3.11+ (developed on 3.14; note that PyTorch's tracing path
warns on 3.14+).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the pretrained models and a sample clip:

```bash
pip install gdown
gdown -O "models/football-player-detection.pt" "https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q"
gdown -O "models/football-pitch-detection.pt"  "https://drive.google.com/uc?id=1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"
gdown -O "models/football-ball-detection.pt"   "https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
gdown -O "data/2e57b9_0.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"
```

Export to OpenVINO (strongly recommended on Intel CPUs — 4.1x here):

```bash
python -c "from ultralytics import YOLO; [YOLO(f'models/football-{m}-detection.pt').export(format='openvino', imgsz=640) for m in ('player','ball','pitch')]"
```

Then point `models:` in `config/config.yaml` at the `_openvino_model/`
directories.

## Usage

```python
from src.utils.config import load_config
from src.perception import pipeline
from src.events.possession import compute_possession, extract_events, summarise

cfg = load_config("config/config.yaml")
df = pipeline.run("data/2e57b9_0.mp4", cfg)      # cached after first run

possession = compute_possession(
    df, cfg.possession.max_distance_cm, cfg.possession.hysteresis_frames
)
summarise(possession, extract_events(possession))
```

All thresholds live in `config/config.yaml`. Nothing in `src/` hardcodes a
tunable value, and the cache key is a hash of the video plus the relevant
config sections — change a threshold and the cache invalidates itself.

```bash
python -m pytest tests/ -v
```

---

## Roadmap

- [ ] Annotated output video: team-coloured boxes, ball marker, top-down radar
- [ ] CLI (`python -m src.cli analyze --video ...`)
- [ ] **Hand-labelled ground truth** for 60–90 s, and possession accuracy
      measured against it — currently the largest gap
- [ ] Ball trajectory interpolation, to recover long passes
- [ ] Native-frame-rate tracking with sampled downstream processing
- [ ] Shot detection (ball velocity vector toward goal mouth)
- [ ] Benchmark on degraded footage: lower resolution, fixed single camera,
      non-broadcast framing

---

## Attribution and licensing

Pretrained detection and pitch-keypoint models, the pitch coordinate
configuration, and the sample footage come from
[roboflow/sports](https://github.com/roboflow/sports). Sample clips originate
from the DFL Bundesliga Data Shootout dataset.

**Ultralytics YOLO is AGPL-3.0 licensed.** That is fine for this repository and
for evaluation, but commercial deployment requires either an Ultralytics
commercial licence or substituting a non-AGPL detector. Worth raising early in
any client conversation rather than discovering it at integration time.

`supervision`'s `ByteTrack` is deprecated as of 0.28 and scheduled for removal
in 0.31; the version in `requirements.txt` is pinned accordingly. Migration is
outstanding.