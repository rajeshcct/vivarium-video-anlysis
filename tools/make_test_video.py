"""Make test videos with KNOWN moving/still periods, so the tracker can be checked against the truth.

1) Synthetic video (a dark mouse-shaped blob that moves, rests and twitches):
    python tools/make_test_video.py synthetic --out samples/synthetic.mp4

2) Real mouse video with frozen sections (the picture is held still for those seconds):
    python tools/make_test_video.py freeze --input samples/test_mouse.mp4 --out samples/frozen.mp4 --freeze 20:35 --freeze 60:80

Each run also writes <out>.truth.csv (start_s, end_s, state). Check the tracker with:
    python tools/compare_truth.py --truth samples/synthetic.truth.csv --timeline output/<session>/timeline.csv
"""
import argparse
import csv
import math
import random
from pathlib import Path

import cv2
import numpy as np

W, H = 640, 480

# (kind, seconds). "twitch" = a tiny 0.4 s movement that should NOT count as moving with min_move_sec = 1.
SCHEDULE = [("moving", 10), ("still", 15), ("moving", 15), ("twitch", 0.4), ("still", 10),
            ("moving", 8), ("still", 20), ("moving", 12), ("still", 10)]


def synthetic(out, fps, speed, seed):
    random.seed(seed)
    np.random.seed(seed)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    x, y, ang = W / 2, H / 2, random.uniform(0, 6.28)
    truth, t = [], 0.0
    for kind, dur in SCHEDULE:
        truth.append((round(t, 2), round(t + dur, 2), "moving" if kind == "moving" else "still"))
        for _ in range(int(round(dur * fps))):
            if kind == "moving":
                ang += random.uniform(-0.3, 0.3)
                x += math.cos(ang) * speed / fps
                y += math.sin(ang) * speed / fps
                if not 60 < x < W - 60:
                    ang = math.pi - ang
                    x = min(max(x, 60), W - 60)
                if not 60 < y < H - 60:
                    ang = -ang
                    y = min(max(y, 60), H - 60)
            elif kind == "twitch":
                x += 10 / (dur * fps)                    # about 10 px in total
            img = np.full((H, W, 3), 200, np.uint8)
            img = np.clip(img + np.random.normal(0, 3, img.shape), 0, 255).astype(np.uint8)
            cv2.ellipse(img, (int(x), int(y)), (32, 15), math.degrees(ang), 0, 360, (60, 60, 60), -1)
            tx, ty = x - math.cos(ang) * 32, y - math.sin(ang) * 32
            cv2.line(img, (int(tx), int(ty)), (int(tx - math.cos(ang) * 25), int(ty - math.sin(ang) * 25)),
                     (90, 90, 90), 2)
            writer.write(img)
        t += dur
    writer.release()
    return truth


def freeze(src, out, intervals):
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    held, n = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = n / fps
        n += 1
        if any(a <= t < b for a, b in intervals):
            if held is None:
                held = frame.copy()
            writer.write(held)
        else:
            held = None
            writer.write(frame)
    writer.release()
    cap.release()
    return [(a, b, "still") for a, b in intervals]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("synthetic")
    s.add_argument("--out", default="samples/synthetic.mp4")
    s.add_argument("--fps", type=float, default=15)
    s.add_argument("--speed", type=float, default=90, help="px/s while moving")
    s.add_argument("--seed", type=int, default=1)
    f = sub.add_parser("freeze")
    f.add_argument("--input", required=True)
    f.add_argument("--out", default="samples/frozen.mp4")
    f.add_argument("--freeze", action="append", required=True, metavar="START:END",
                   help="seconds to freeze, e.g. 20:35 (repeatable)")
    a = ap.parse_args()

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    if a.mode == "synthetic":
        truth = synthetic(a.out, a.fps, a.speed, a.seed)
    else:
        iv = [tuple(float(v) for v in x.split(":")) for x in a.freeze]
        truth = freeze(a.input, a.out, iv)

    tp = Path(a.out).with_suffix(".truth.csv")
    with open(tp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["start_s", "end_s", "state"])
        w.writerows(truth)
    print(f"Video : {a.out}\nTruth : {tp}")


if __name__ == "__main__":
    main()
