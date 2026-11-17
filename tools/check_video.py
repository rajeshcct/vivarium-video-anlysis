"""Test a frame source from the config: opens it, reads frames, reports real FPS and size.

    python tools/check_video.py                       # uses config.yaml
    python tools/check_video.py --config config_pi.yaml
    python tools/check_video.py --set source.type=usb --show

Saves two images to output/:
  first_frame.jpg            – raw full-capture-resolution frame
                               USE THIS to measure ROI coordinates (open in Paint / GIMP,
                               hover the mouse to read x,y pixel coords)
  first_frame_annotated.jpg  – same frame with ROI box + size annotations drawn on it
                               USE THIS to verify the ROI is in the right place

Also prints a scale report so you know exactly what coordinate space every
threshold (min_area, speed_threshold, max_match_dist ...) lives in.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.config import load_config, validate
from src.source import make_source


def _draw_info(frame, cfg):
    """Draw ROI box and size annotations on a copy of frame."""
    v = cfg["video"]
    h, w = frame.shape[:2]
    img = frame.copy()

    roi = v["roi"]
    pw  = v["process_width"]

    if roi:
        rx, ry, rw, rh = roi
        # clamp to actual frame
        rx = min(rx, w - 1); ry = min(ry, h - 1)
        rw = min(rw, w - rx); rh = min(rh, h - ry)
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
        cv2.putText(img, f"ROI {rw}x{rh}px  (green box)",
                    (rx + 4, ry + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        ph = max(1, round(rh * pw / rw)) if pw else rh
        info = (f"Capture {w}x{h}  |  ROI {rw}x{rh} @ ({rx},{ry})  |  "
                f"Process {pw or rw}x{ph}  |  "
                f"Scale {rw/(pw or rw):.2f}x  {rh/ph:.2f}y")
    else:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 200, 255), 2)
        cv2.putText(img, "No ROI – full frame used  (yellow box)",
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)
        ph = max(1, round(h * pw / w)) if pw else h
        info = (f"Capture {w}x{h}  |  No ROI  |  Process {pw or w}x{ph}  |  "
                f"Scale {w/(pw or w):.2f}x  {h/ph:.2f}y")

    cv2.putText(img, info, (8, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
    return img, info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--frames", type=int, default=60,
                    help="frames to read for the throughput test (default 60)")
    ap.add_argument("--show", action="store_true",
                    help="open the annotated frame in a window (needs a display)")
    args = ap.parse_args()

    cfg = load_config(args.config if Path(args.config).is_file() else None, args.set)
    validate(cfg)
    src = make_source(cfg)

    print(f"\nSource type  : {cfg['source']['type']}")
    print(f"Effective FPS: {src.fps:.2f}")

    ok, frame, t = src.read()
    if not ok:
        print("ERROR: Could not read a single frame")
        sys.exit(1)
    h, w = frame.shape[:2]

    # ── Coordinate / scale report ─────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"CAPTURE SIZE : {w} x {h}  px")
    print(f"  → video.roi coordinates must be in THIS space")
    print(f"  → use first_frame.jpg to measure x,y with Paint/GIMP pixel tool")

    v = cfg["video"]
    roi = v["roi"]
    pw  = v["process_width"]

    if roi:
        rx, ry, rw, rh = roi
        rw_clamped = min(rw, w - rx)
        rh_clamped = min(rh, h - ry)
        if rw_clamped != rw or rh_clamped != rh:
            print(f"  ⚠  ROI was clamped: requested {rw}x{rh}, "
                  f"clamped to {rw_clamped}x{rh_clamped} (exceeds frame edge)")
            rw, rh = rw_clamped, rh_clamped
        ph = max(1, round(rh * pw / rw)) if pw else rh
        scale_x = rw / (pw or rw)
        scale_y = rh / ph
        print(f"\nROI          : x={rx}  y={ry}  w={rw}  h={rh}")
        print(f"PROCESS SIZE : {pw or rw} x {ph}  px")
        print(f"  → ALL thresholds below are in PROCESS pixels:")
        print(f"     detection.min_area, max_area, close_kernel, max_match_dist")
        print(f"     movement.speed_threshold, jitter_px")
        print(f"Scale        : {scale_x:.3f} capture-px per process-px (x)")
        print(f"               {scale_y:.3f} capture-px per process-px (y)")
        if abs(scale_x - scale_y) > 0.05:
            print(f"  ⚠  X and Y scales differ by "
                  f"{abs(scale_x-scale_y)/max(scale_x,scale_y)*100:.1f}% — "
                  f"blobs will appear squished. "
                  f"Consider adjusting process_width to match the ROI aspect ratio.")
        else:
            print(f"  ✓  Scales match within 5% — no blob distortion")
    else:
        ph = max(1, round(h * pw / w)) if pw else h
        print(f"\nROI          : none  (full frame used)")
        print(f"PROCESS SIZE : {pw or w} x {ph}  px")
    print(f"{'─'*60}\n")

    # ── Speed test ────────────────────────────────────────────────────────
    first = frame
    t_start, n = time.time(), 1
    while n < args.frames:
        ok, frame, t = src.read()
        if not ok:
            break
        n += 1
    elapsed = time.time() - t_start
    print(f"Speed test   : {n} frames  |  last t={t:.2f}s  |  "
          f"{n / max(elapsed, 1e-6):.1f} frames/s throughput")

    # ── Save images ───────────────────────────────────────────────────────
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)

    raw_path = out / "first_frame.jpg"
    ann_path = out / "first_frame_annotated.jpg"

    annotated, info_line = _draw_info(first, cfg)
    cv2.imwrite(str(raw_path),  first)
    cv2.imwrite(str(ann_path),  annotated)

    print(f"\nSaved (raw)       : {raw_path}")
    print(f"  Open in Paint / GIMP, hover pixel to read (x,y) for roi coordinates")
    print(f"Saved (annotated) : {ann_path}")
    print(f"  Green box = ROI, yellow text = size info. Verify it covers the cage floor.")
    print(f"\nInfo line: {info_line}")

    if args.show:
        cv2.imshow("check_video — press any key", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    src.release()


if __name__ == "__main__":
    main()
