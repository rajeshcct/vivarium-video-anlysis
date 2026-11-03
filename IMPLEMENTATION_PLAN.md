# Vivarium Video: Mouse Movement Tracker (Implementation Plan)

## 1. Goal

Detect whether a mouse is **moving or still** from video, and report how much time it spends in each state after drug treatment. Behaviour classification (grooming, rearing, etc.) is **out of scope** for this stage.

**Principle:** nothing is hard-coded. Every setting (source, FPS, resolution, ROI, thresholds, buffers, output) lives in `config.yaml`. The same code runs on a laptop with a sample video and on a Raspberry Pi with a camera by changing the config only.

## 2. Scope

| In scope | Out of scope (later) |
|---|---|
| Video file, USB camera, Pi camera, RTSP input | Pose estimation (DeepLabCut/SLEAP) |
| Mouse centroid tracking with OpenCV | Behaviour classification (SimBA) |
| Moving / still timeline, bouts, distance, speed | Multi-animal tracking |
| Motion-triggered clips with pre/post buffer | GPU training |
| CSV outputs with the settings used | Dashboard, SQLite (optional later) |

## 3. Project structure

```
Vivarium -video/
├── IMPLEMENTATION_PLAN.md
├── config.yaml            # all settings (laptop / sample video)
├── config_pi.yaml         # Pi camera overrides
├── requirements.txt
├── main.py                # entry point (single run or batch)
├── src/
│   ├── config.py          # defaults, load, overrides, validation
│   ├── source.py          # file / usb / picamera / rtsp behind one interface
│   ├── detector.py        # background subtraction, centroid, speed, overlay
│   ├── timeline.py        # moving/still bouts and summary measures
│   ├── recorder.py        # pre/post-buffer clip saving
│   ├── output.py          # per-session output files + config copy
│   └── graphs.py          # movement graphs (speed, activity, per-minute, distance, path, heatmap)
├── tools/
│   ├── check_video.py     # test any source from the config
│   ├── make_test_video.py # synthetic / frozen test videos with known truth
│   ├── compare_truth.py   # compare a timeline with the truth file
│   ├── summarize.py       # merge all session summaries into one table
│   └── make_graphs.py     # redo the graphs for a finished session
├── README.md
├── samples/               # put test videos here
└── output/                # one sub-folder per session
```

## 4. Output per session (`output/<session_id>/`)

| File | Content |
|---|---|
| `config_used.yaml` | exact settings of this run (reproducibility) |
| `frame_data.csv` | per frame: time, x, y, found, speed, moving |
| `timeline.csv` | start, end, duration, state (moving / still) |
| `summary.csv` | session time, moving/still time, % moving, bouts, longest still, distance, mean speed, % frames with mouse found |
| `clips/` and `clips.csv` | motion-triggered clips and their start/end times |
| `overlay.mp4` | optional video with tracked dot and MOVING/STILL label |
| `speed_timeline.png`, `activity_bar.png`, `moving_per_minute.png`, `distance_cumulative.png`, `trajectory.png`, `heatmap.png` | movement graphs (automatic when `output.auto_graphs` is true) |

## 5. Phases

Status legend: `[x]` code written, `[ ]` to do. Code is written but **not yet run**; each phase has an acceptance test to confirm it.

### Phase 0: Environment and skeleton (2 days)
- [x] Folder structure, `requirements.txt`
- [ ] Install Python 3.10/3.11, create venv, `pip install -r requirements.txt`
- [ ] Put 1-3 sample videos in `samples/`

**Done when:** `python tools/check_video.py` prints FPS and frame size for the sample video.

### Phase 1: Config system (2 days)
- [x] `src/config.py`: defaults, YAML loading, `--set key=value` overrides, validation, unknown-key detection (catches typos)
- [x] Config copy saved with every result

**Done when:** changing a value in `config.yaml` changes behaviour with no code edit, and a bad value or typo gives a clear error.

### Phase 2: Source abstraction (3-4 days)
- [x] `FileSource`, `UsbSource`, `RtspSource`, `PiCameraSource` behind one `read()` returning `(ok, frame, timestamp)`
- [x] FPS handling: `auto` or a number (frames skipped when target is lower than the source)
- [x] Rotation, `loop`, auto-reconnect for live sources
- [ ] Test file source, then USB webcam

**Done when:** the same `main.py` processes a video file and a laptop webcam by changing only `source.type`.

### Phase 3: Movement detection (1 week)
- [x] Three methods: `mog2` (adaptive), `fixed_background` (empty-cage image), `frame_diff`
- [x] Centroid of largest blob, speed over a short window, ROI, warm-up period
- [x] Overlay video / preview window
- [ ] Tune `min_area`, `var_threshold`, `speed_threshold` on 3-5 videos

**Done when:** you watch the overlay and agree with MOVING/STILL on most of the video, and have listed missed and false detections.

### Phase 4: Timeline and measures (3-4 days)
- [x] Frame states merged into bouts (`min_state_sec` removes flicker)
- [x] `timeline.csv` and `summary.csv`, cm conversion when `px_per_cm` is set
- [ ] Check totals against the video by eye

**Done when:** totals match what you see in the video.

### Phase 5: Clip recorder (3-4 days)
- [x] Pre-buffer in memory, post-buffer after stillness, continuous mode, `clips.csv`
- [ ] Check that clips include the seconds before movement starts

**Done when:** clip log matches the timeline.

### Phase 6: Batch and summary (3 days)
- [x] `python main.py --input samples/` processes all videos
- [x] `tools/summarize.py` merges all summaries into `output/all_summaries.csv`
- [ ] Run 5 videos in one command

**Done when:** one comparison table for 5 videos.

### Phase 7: Pi camera and live mode (1 week)
- [x] `PiCameraSource` using Picamera2, `config_pi.yaml`, manual focus / fixed exposure / fixed white balance from the `picamera` config section
- [x] Health log every minute: fps, frame gaps, CPU temperature, free disk
- [ ] Install on the Pi, test live: CPU, temperature, dropped frames
- [ ] 1+ hour run; also test with the camera pointed at a screen playing a mouse video
- [x] Clean stop with Ctrl+C, SIGTERM or `max_duration_sec`

**Done when:** the Pi runs 1+ hour without crashes and its results resemble the laptop results.

### Phase 8: Reliability and handover (3-4 days)
- [x] Reconnect on camera loss, free-disk check before and during recording, SIGTERM-safe stop
- [x] Clock sync notes in README (section 7)
- [ ] Freeze tested settings as `config_v1.yaml` (copy `config_used.yaml` after tuning on real footage)
- [x] `README.md` written (install, run, outputs, tuning, Pi, study rules, limits)

**Done when:** someone else can install it and reproduce the same results.

## 6. Timeline

| Phase | Duration |
|---|---|
| 0. Environment | 2 days |
| 1. Config | 2 days |
| 2. Sources | 3-4 days |
| 3. Detection | 1 week |
| 4. Timeline | 3-4 days |
| 5. Recorder | 3-4 days |
| 6. Batch | 3 days |
| 7. Pi and live | 1 week |
| 8. Handover | 3-4 days |
| **Total** | **about 5 weeks** (Phases 0-6 need only a laptop) |

## 7. Known limits (tell the client)

- Detects **gross movement only**. Grooming, tremor, and breathing may count as "still".
- A motionless mouse can be absorbed into an adaptive background. If that happens, use `fixed_background` with an empty-cage image.
- Needs a fixed camera, constant lighting, and good contrast between mouse and floor.
- Speed is in pixels per second at the processing resolution. Set `px_per_cm` (measured at that resolution) to get cm and cm/s.
- Accuracy must be checked against a human scorer on footage from the real setup before any claim is made.
- For drug studies: record **continuously** in real experiments. A motion trigger alone can lose data when a drug sedates the animal. Triggered clips are a storage-saving option, and the timeline logs still periods either way.
- Use blinded session IDs (`treatment_code`), not drug names, in file names.

## 8. Quick start

```bash
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt

python tools/check_video.py                      # test the source from config.yaml
python main.py                                   # run one session
python main.py --input samples/                  # run all videos in a folder
python main.py --set detection.min_area=200 --set movement.speed_threshold=12
python main.py --config config_pi.yaml           # on the Pi
python tools/summarize.py                        # merge all summaries
```

Pi notes: install Picamera2 with `sudo apt install python3-picamera2`, create the venv with `--system-site-packages`, and use `opencv-python-headless`.
