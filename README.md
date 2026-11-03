# Vivarium Video: Mouse Movement Tracker

Measures whether a mouse is **moving or still** from video, after drug treatment. It works from a video file, USB camera, Raspberry Pi camera, or RTSP stream. Every setting is in `config.yaml`.

It does **not** recognise behaviours (grooming, rearing, ...) and does not know that something is a mouse: it tracks whatever differs from the empty cage. See "Known limits".

## 1. Install

Laptop (Windows / Mac / Linux):
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

Raspberry Pi 5 (Bookworm):
```bash
sudo apt install -y python3-picamera2 python3-opencv python3-yaml python3-numpy
python3 -m venv --system-site-packages venv
source venv/bin/activate
```
Use the system OpenCV on the Pi, and do not pip-install `opencv-python` there.

## 2. Run

```bash
python tools/check_video.py --show               # test the source (FPS, size, first frame)
python main.py                                   # one session using config.yaml
python main.py --input samples/                  # every video in a folder
python main.py --config config_pi.yaml           # Pi camera
python main.py --set output.overlay_video=true   # also save a video with MOVING / STILL labels
python main.py --set detection.min_area=200 --set movement.speed_threshold=12
python tools/summarize.py                        # merge all sessions into output/all_summaries.csv
```
Stop a live run with Ctrl+C. Files are closed properly.

## 3. Outputs (`output/<session_id>/`)

| File | Content |
|---|---|
| `summary.csv` | session time, moving / still time, % moving, bouts, longest still, distance, mean speed, % frames with mouse found |
| `timeline.csv` | start, end, duration, state (moving / still) |
| `frame_data.csv` | per frame: time, x, y, found, speed, moving |
| `clips/`, `clips.csv` | motion-triggered clips and their times |
| `overlay.mp4` | optional check video |
| `speed_timeline.png` | speed against time, speed threshold, moving periods shaded |
| `activity_bar.png` | one bar across the session: moving / still, and % moving |
| `moving_per_minute.png` | % of each minute spent moving (compare treated and control animals) |
| `distance_cumulative.png` | distance travelled so far; ends at the `summary.csv` distance |
| `trajectory.png` | whole-session path in the cage, coloured by time, start and end marked |
| `heatmap.png` | where in the cage the mouse spent time |
| `config_used.yaml` | exact settings of this run |

The graphs are made automatically at the end of a session (`output.auto_graphs: true`) from `frame_data.csv` and `timeline.csv`. No video is processed, so it takes a few seconds even for a long session. They are in pixels until `movement.px_per_cm` is set, then in cm. The path uses the tracked centre point only, and gaps appear where the mouse was not detected. To redo them later:
```bash
python tools/make_graphs.py output/<session_folder>
python tools/make_graphs.py output/<session_folder> --px-per-cm 6.5   # draw in cm without re-running the session
```
Install matplotlib once: laptop `pip install matplotlib`, Pi `sudo apt install -y python3-matplotlib`.

## 4. Main settings

| Setting | Meaning |
|---|---|
| `source.type` | `file`, `usb`, `picamera`, `rtsp` |
| `video.fps` | `auto` or a number |
| `video.roi` | `[x, y, w, h]` to analyse only the cage |
| `detection.method` | `mog2`, `fixed_background` (empty-cage photo), `frame_diff` |
| `detection.min_area` | smallest blob accepted as the mouse |
| `movement.speed_threshold` | px/s counted as moving |
| `movement.min_move_sec` | movement shorter than this is ignored |
| `movement.min_still_sec` | a pause shorter than this does not split a bout |
| `movement.px_per_cm` | set to get cm and cm/s |
| `recording.pre_buffer_sec` / `post_buffer_sec` | clip seconds before / after movement |
| `recording.continuous` | `true` records the whole session |
| `output.auto_graphs` | `true` makes the movement graphs at the end of a session |

Missing settings use defaults. A mistyped setting name gives an error.

## 5. Tuning on a new setup

0. Set `video.roi` so only the cage floor is analysed (anything moving outside the cage can be tracked as the mouse):
   ```bash
   python tools/check_video.py --config config_pi.yaml     # saves output/first_frame.jpg
   python tools/pick_roi.py output/first_frame.jpg         # drag a box around the cage floor, press Enter
   ```
   Paste the printed `roi: [x, y, w, h]` into the config. With a smaller analysed area the mouse looks bigger in pixels, so the tool also prints new values for `detection.min_area` and `movement.speed_threshold`.
1. Run with `output.overlay_video=true` and watch the video.
2. `mouse_found_pct` in `summary.csv` should be above 80%. If not: adjust `min_area`, `roi`, `method`, lighting.
3. If a resting mouse disappears or the label is wrong during rest: use `fixed_background` with a picture of the empty cage (`detection.background_image`). With no empty cage, build the picture from clips of an earlier session, where the mouse moves around:
   ```bash
   python tools/make_background.py output/<session_folder>/clips --out background.jpg
   ```
   Open `background.jpg`: the mouse must not be visible (a faint ghost means it sat still in that spot for most of the frames: add more clips). Make a new one if the camera, light or cage changes.
4. Tune `speed_threshold`, `min_move_sec`, `min_still_sec` against a human scorer or a test video with known truth (`tools/make_test_video.py`, `tools/compare_truth.py`).
5. Copy the final `config_used.yaml` to `config_v1.yaml` and use it unchanged for the whole study.

## 6. Raspberry Pi camera (Camera Module 3)

- Settings are in the `picamera` section: manual focus, fixed exposure, fixed white balance. Auto focus and auto exposure create false motion.
- `lens_position` = 1 / distance in metres (about 3.3 for 30 cm). `exposure_us`: use multiples of 10000 on 50 Hz mains to avoid light flicker.
- Standard Module 3 needs visible light. For dark-phase work use the NoIR version with an 850 nm IR lamp.
- Mount the camera rigidly. Vibration looks like movement.
- Use an active cooler and an SSD (set `recording.clip_dir` and `output.dir` to the SSD).
- Every `logging.health_interval_sec` seconds the log shows fps, frame gaps, CPU temperature and free disk. The run stops safely if disk space falls below `recording.min_free_gb`.

## 6b. USB webcam on the Pi (instead of the Pi camera)

Use `config_pi_webcam.yaml`. The Pi camera mode (`config_pi.yaml`) is not changed.

1. `sudo apt install -y v4l-utils`, then find the webcam with `ls -l /dev/v4l/by-id/`. Put the path ending in `-video-index0` in `source.usb_device`. Do not rely on `usb_index: 0`: the Pi's own camera and decoder also create `/dev/video*` devices.
2. `python tools/check_video.py --config config_pi_webcam.yaml` shows the real size and frame rate and saves `output/first_frame.jpg`. A webcam that cannot do 1280x720 is reported in the log; `video.roi` must use the size it really gives.
3. Set the ROI again with `tools/pick_roi.py` (a new camera means new numbers), and use the `min_area` and `speed_threshold` it prints.
4. Fix the image for an experiment in the `usbcam` section (exposure, focus, white balance). With the settings left at `null` the webcam adjusts itself, which can look like movement. Not every webcam accepts every control. `v4l2-ctl -d <device> --list-ctrls` lists what yours supports.
5. Run: `python main.py --config config_pi_webcam.yaml`.

MJPG (`source.usb_fourcc`) is the default because most webcams only reach full speed at HD in that format. A webcam is decoded on the CPU, so check the `health:` log line for fps and CPU temperature on a long run.

## 7. Several units: clock sync

Use the same time on all devices so sessions can be compared by time of day:
```bash
timedatectl status             # should show "System clock synchronized: yes"
sudo timedatectl set-ntp true
```
For a room without internet, run `chrony` on one unit as the time server for the others. Set the time zone with `sudo timedatectl set-timezone Asia/Kolkata`.

## 8. Study rules

- Same config, camera position, and lighting for every animal and group.
- Use blinded codes in `session.treatment_code`, not drug names.
- Include vehicle controls.
- For real experiments record **continuously** (`recording.continuous: true`). A drug that sedates the animal may never trigger a motion-based clip. The timeline logs still periods either way.
- Back up the raw videos and `output/` folders to a second location.

## 9. Known limits

- Detects gross movement only. Grooming, tremor and breathing may count as still.
- Does not identify the animal: a hand, a shadow or a moving object in view can be tracked instead. Use `roi` and keep the scene clean.
- An adaptive background (`mog2`) can absorb a motionless mouse. Use `fixed_background` if this happens.
- Speed is in pixels per second at the processing size until `px_per_cm` is set.
- Accuracy must be checked against a human scorer on footage from the real setup before claims are made.

## 10. Project layout

```
config.yaml, config_pi.yaml, config_pi_webcam.yaml   settings
main.py                       entry point
src/                          config, source, detector, timeline, recorder, output, graphs, usbcam
tools/                        check_video, pick_roi, make_background, summarize, make_test_video, compare_truth, make_graphs
samples/, output/             test videos, results
IMPLEMENTATION_PLAN.md        plan and phase status
```
