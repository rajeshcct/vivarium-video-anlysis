"""Make an empty-cage background picture from videos that already contain the mouse.

The mouse moves around, so for every pixel the MIDDLE value over many frames is the empty floor.
Use it as detection.background_image with detection.method = fixed_background.

    python tools/make_background.py output/<session_folder>/clips
    python tools/make_background.py clip1.mp4 clip2.mp4 --frames 200 --out background.jpg

Open background.jpg afterwards: the mouse must NOT be visible. If a faint "ghost" mouse is left, the mouse
sat in that place for more than half of the sampled frames: give it more videos (other sessions, clips
where it moved a lot) or more --frames, then check again.

Use videos from the same camera position and lighting as the experiment. Make a new one if the camera
is moved, the light changes or the cage is changed.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def collect(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT)
        elif p.is_file():
            files.append(p)
        else:
            sys.exit(f"Not found: {p}")
    if not files:
        sys.exit("No video files found.")
    return files


def main():
    ap = argparse.ArgumentParser(description="Build a background picture from videos with a moving mouse")
    ap.add_argument("videos", nargs="+", help="video files and/or folders of videos")
    ap.add_argument("--frames", type=int, default=120, help="how many frames to sample (default 120)")
    ap.add_argument("--scale", type=float, default=0.5, help="work at this fraction of the size to save RAM (default 0.5)")
    ap.add_argument("--out", default="background.jpg")
    a = ap.parse_args()

    files = collect(a.videos)
    infos, size = [], None
    for f in files:
        cap = cv2.VideoCapture(str(f))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if n <= 0 or w <= 0:
            print(f"Skipped (cannot read): {f}")
            continue
        if size is None:
            size = (w, h)
        if (w, h) != size:
            print(f"Skipped (size {w}x{h}, expected {size[0]}x{size[1]}): {f}")
            continue
        infos.append((f, n))
    if not infos:
        sys.exit("No usable video.")

    total = sum(n for _, n in infos)
    want = min(a.frames, total)
    W, H = size
    sw, sh = max(1, int(W * a.scale)), max(1, int(H * a.scale))
    print(f"{len(infos)} video(s), {total} frames in total, sampling {want} frames at {sw}x{sh}")

    # sample evenly over all frames of all videos
    picks = np.linspace(0, total - 1, want).astype(int)
    stack, offset = [], 0
    for f, n in infos:
        mine = [int(p - offset) for p in picks if offset <= p < offset + n]
        offset += n
        if not mine:
            continue
        cap = cv2.VideoCapture(str(f))
        for idx in mine:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if ok:
                stack.append(cv2.resize(fr, (sw, sh), interpolation=cv2.INTER_AREA))
        cap.release()
    if len(stack) < 10:
        sys.exit(f"Only {len(stack)} frames could be read: too few to build a background.")

    bg = np.median(np.stack(stack), axis=0).astype(np.uint8)
    bg = cv2.resize(bg, (W, H), interpolation=cv2.INTER_LINEAR)       # back to the camera's real size
    cv2.imwrite(a.out, bg)
    print(f"Saved {a.out} ({W}x{H}) from {len(stack)} frames.")
    print("Open it now: the mouse must not be visible. Then in config_pi.yaml:")
    print("  detection:\n    method: fixed_background\n    background_image: " + a.out)


if __name__ == "__main__":
    main()
