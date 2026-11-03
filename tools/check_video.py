"""Test a frame source from the config: opens it, reads frames, reports real FPS and size.

    python tools/check_video.py                       # uses config.yaml
    python tools/check_video.py --config config_pi.yaml
    python tools/check_video.py --set source.type=usb --show
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.config import load_config, validate
from src.source import make_source


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--frames", type=int, default=60, help="frames to read for the speed test")
    ap.add_argument("--show", action="store_true", help="show the first frame in a window")
    args = ap.parse_args()

    cfg = load_config(args.config if Path(args.config).is_file() else None, args.set)
    validate(cfg)
    src = make_source(cfg)
    print(f"Source type : {cfg['source']['type']}")
    print(f"Effective FPS: {src.fps:.2f}")

    ok, frame, t = src.read()
    if not ok:
        print("Could not read a frame")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"Frame size  : {w} x {h}")

    first = frame
    t_start, n = time.time(), 1
    while n < args.frames:
        ok, frame, t = src.read()
        if not ok:
            break
        n += 1
    elapsed = time.time() - t_start
    print(f"Read {n} frames, last timestamp {t:.2f}s, reading speed {n / max(elapsed, 1e-6):.1f} frames/s")

    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    jpg = out / "first_frame.jpg"
    cv2.imwrite(str(jpg), first)
    print(f"First frame saved to {jpg}")
    if args.show:
        cv2.imshow("First frame (press any key)", first)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    src.release()


if __name__ == "__main__":
    main()
