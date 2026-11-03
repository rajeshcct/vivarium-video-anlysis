"""Find the numbers for video.roi by drawing a box on a picture of the cage.

    python tools/check_video.py --config config_pi.yaml        # saves output/first_frame.jpg
    python tools/pick_roi.py output/first_frame.jpg            # a window opens: drag a box, press Enter

    python tools/pick_roi.py output/first_frame.jpg --roi 570,150,490,480   # no window: only draws that box, to check it

The picture can be any image or a video (its first frame is used). You can copy first_frame.jpg to your
laptop and run this there: the numbers are pixels of the picture, the same pixels video.roi uses.
Drag the box around the cage FLOOR where the mouse can walk, with a small margin, and leave out walls,
lids, cables and anything else that moves or changes.

It prints the line for config_pi.yaml and saves roi_preview.jpg next to the picture.
"""
import argparse
import sys
from pathlib import Path

import cv2


def load_picture(path):
    p = Path(path)
    if not p.is_file():
        sys.exit(f"File not found: {p}")
    img = cv2.imread(str(p))
    if img is None:                                   # not an image: try it as a video
        cap = cv2.VideoCapture(str(p))
        ok, img = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Cannot read a picture or a video frame from {p}")
    return img


def pick_with_window(img):
    h, w = img.shape[:2]
    scale = min(1.0, 1000.0 / w)                      # fit small screens, numbers are converted back
    shown = cv2.resize(img, None, fx=scale, fy=scale) if scale < 1 else img
    r = cv2.selectROI("Drag a box around the cage floor, then press Enter (c = cancel)",
                      shown, showCrosshair=False, fromCenter=False)
    cv2.destroyAllWindows()
    x, y, rw, rh = (int(round(v / scale)) for v in r)
    return x, y, rw, rh


def main():
    ap = argparse.ArgumentParser(description="Pick video.roi by drawing a box on a picture")
    ap.add_argument("picture", help="image (e.g. output/first_frame.jpg) or video file")
    ap.add_argument("--roi", help="x,y,w,h : skip the window and only check these numbers")
    ap.add_argument("--min-area", type=float, default=150, help="your current detection.min_area")
    ap.add_argument("--speed-threshold", type=float, default=8, help="your current movement.speed_threshold")
    a = ap.parse_args()

    img = load_picture(a.picture)
    H, W = img.shape[:2]
    print(f"Picture size: {W} x {H} pixels")

    if a.roi:
        try:
            x, y, rw, rh = (int(v) for v in a.roi.split(","))
        except ValueError:
            sys.exit("--roi must be four whole numbers: x,y,w,h")
    else:
        x, y, rw, rh = pick_with_window(img)
    if rw <= 0 or rh <= 0:
        sys.exit("No box selected.")
    if x < 0 or y < 0 or x + rw > W or y + rh > H:
        sys.exit(f"The box [{x}, {y}, {rw}, {rh}] is not inside the {W} x {H} picture.")

    preview = img.copy()
    cv2.rectangle(preview, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
    cv2.putText(preview, f"roi: [{x}, {y}, {rw}, {rh}]", (x + 6, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    out = Path(a.picture).with_name("roi_preview.jpg")
    cv2.imwrite(str(out), preview)

    f = W / rw       # the analysis image is now the box, not the whole frame, so the mouse looks bigger
    print(f"\nPut this in config_pi.yaml (and config.yaml):\n\nvideo:\n  roi: [{x}, {y}, {rw}, {rh}]\n")
    print(f"Preview saved to {out}  (open it and check the green box covers the whole cage floor)\n")
    print(f"The mouse will look {f:.1f}x bigger in the analysis image, so keep the same sensitivity with:")
    print(f"  detection.min_area: {a.min_area * f * f:.0f}        (was {a.min_area:g})")
    print(f"  movement.speed_threshold: {a.speed_threshold * f:.0f}     (was {a.speed_threshold:g})")
    print("Tune both again after a short test run.")


if __name__ == "__main__":
    main()
