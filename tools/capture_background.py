"""Capture a single frame from the configured source and save it as the
fixed_background reference image.

Run this ONCE with the cage empty (no mice) before starting real sessions:

    python tools/capture_background.py                   # uses config.yaml
    python tools/capture_background.py --config config_pi.yaml

The image is saved to the path set in detection.background_image (or
samples/background.jpg by default).  Then set
    detection.background_image: samples/background_pi.jpg
in your config and you are ready to run with method: fixed_background.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from src.config import load_config, validate
from src.source import make_source


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--out", default=None,
                    help="Output path (default: value of detection.background_image, "
                         "or samples/background.jpg)")
    ap.add_argument("--warmup", type=int, default=5,
                    help="Number of frames to discard before capturing (lets exposure settle)")
    args = ap.parse_args()

    # Load config but skip validation of background_image path
    # (it doesn't exist yet — we're creating it)
    cfg = load_config(args.config if Path(args.config).is_file() else None, args.set)

    out_path = args.out or cfg["detection"].get("background_image") or "samples/background.jpg"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    src = make_source(cfg)
    h_cap, w_cap = None, None

    print(f"Warming up — discarding {args.warmup} frames to let exposure/focus settle...")
    for i in range(args.warmup + 1):
        ok, frame, t = src.read()
        if not ok:
            print("ERROR: Could not read a frame from source")
            sys.exit(1)
        if h_cap is None:
            h_cap, w_cap = frame.shape[:2]
        if i < args.warmup:
            print(f"  frame {i+1}/{args.warmup}...")

    src.release()

    cv2.imwrite(str(out_path), frame)

    print(f"\nCapture size : {w_cap} x {h_cap}  px")
    print(f"Saved to     : {out_path.resolve()}")
    print(f"\nNext step:")
    print(f"  Set in your config:")
    print(f"    detection:")
    print(f"      method: fixed_background")
    print(f"      background_image: {out_path}")
    print(f"  Then run:  python main.py --config {args.config}")


if __name__ == "__main__":
    main()

