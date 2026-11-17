"""Tools for turning several short AI-generated clips into one long, loop-friendly test video.

You only generate the clips in Gemini (Veo). This script does everything else:

  1) frames  - save the first and last frame of a clip as PNG. Upload the LAST frame to Gemini as the start
               image of the next clip, and use the FIRST frame of clip 1 as the end image of the final clip.

        python tools/ai_video_tools.py frames clip1.mp4

  2) build   - make all clips the same size / frame rate, join them in the order given, CHECK every join and
               the loop closure (end of the last clip vs start of the first), optionally close the loop with a
               ping-pong, and repeat the result N times.

        python tools/ai_video_tools.py build clip1.mp4 clip2.mp4 clip3.mp4 clip4.mp4 --repeat 10 --out samples/long_test.mp4

Needs only OpenCV and numpy (already in your project).
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

THUMB = (320, 180)          # pictures are compared at this size
CHANGE_LEVEL = 40           # gray-level difference that counts as "changed"


# ---------------------------------------------------------------------------------------------------------
def open_video(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or w <= 0 or h <= 0:
        sys.exit(f"Cannot read size / frame rate of: {path}")
    return cap, fps, w, h, n


def parse_size(text):
    try:
        w, h = (int(v) for v in text.lower().split("x"))
        if w > 0 and h > 0:
            return w, h
    except ValueError:
        pass
    sys.exit("--size must look like 1280x720")


def thumb_gray(frame):
    g = cv2.cvtColor(cv2.resize(frame, THUMB, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (5, 5), 0)


def compare(g1, g2):
    """(% of the picture that changed, brightness difference)"""
    d = cv2.absdiff(g1, g2)
    return float((d > CHANGE_LEVEL).mean() * 100), abs(float(g1.mean()) - float(g2.mean()))


def verdict(changed, limit):
    return "OK" if changed <= limit else "JUMP"


def resampled_frames(path, size, fps):
    """Frames of one clip at the target size and frame rate, keeping the real duration."""
    cap, src_fps, w, h, _ = open_video(path)
    step, out_t, j = 1.0 / fps, 0.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if (w, h) != size:
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        end_t = (j + 1) / src_fps
        while out_t < end_t - 1e-9:
            yield frame
            out_t += step
        j += 1
    cap.release()


# ---------------------------------------------------------------------------------------------------------
def cmd_frames(a):
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for p in a.videos:
        cap, fps, w, h, n = open_video(p)
        first = last = None
        count = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if first is None:
                first = fr
            last = fr
            count += 1
        cap.release()
        if first is None:
            print(f"No frames in {p}")
            continue
        stem = Path(p).stem
        f_path, l_path = out / f"{stem}_first.png", out / f"{stem}_last.png"
        cv2.imwrite(str(f_path), first)
        cv2.imwrite(str(l_path), last)
        print(f"{Path(p).name}: {w}x{h} @ {fps:.2f} fps, {count / fps:.1f} s")
        print(f"   first frame -> {f_path}")
        print(f"   last frame  -> {l_path}")
    print("\nNext clip in Gemini: use the LAST frame of the previous clip as the start image.")
    print("Final clip: also use the FIRST frame of clip 1 as the end image, if your tool has that option.")


def cmd_build(a):
    if a.repeat < 1:
        sys.exit("--repeat must be 1 or more")
    for p in a.videos:
        if not Path(p).is_file():
            sys.exit(f"File not found: {p}")
    size, fps = parse_size(a.size), a.fps
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / f".cycle_tmp_{os.getpid()}.avi"

    print(f"Output: {size[0]}x{size[1]} @ {fps:g} fps")
    cyc = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not cyc.isOpened():
        sys.exit("Cannot create the temporary file (MJPG codec missing in this OpenCV build).")

    firsts, lasts, steps, n = [], [], [], 0
    for p in a.videos:
        first = prev = None
        cnt = 0
        for fr in resampled_frames(p, size, fps):
            cyc.write(fr)
            g = thumb_gray(fr)
            if first is None:
                first = g
            if prev is not None:
                steps.append(compare(prev, g)[0])
            prev = g
            cnt += 1
        if first is None:
            cyc.release()
            tmp.unlink(missing_ok=True)
            sys.exit(f"No frames in {p}")
        firsts.append(first)
        lasts.append(prev)
        n += cnt
        print(f"  {Path(p).name}: {cnt / fps:.1f} s")
    cyc.release()

    # ---- checks -------------------------------------------------------------------------------------
    base = float(np.percentile(steps, 95)) if steps else 0.0
    limit = max(0.3, 1.5 * base + 0.05)
    print(f"\nChecks. Normal frame-to-frame change inside your clips: {base:.2f}% of the picture (95th percentile),")
    print(f"so a join above {limit:.2f}% counts as a jump (mice suddenly in other places):")
    bright_bad = False
    for i in range(len(a.videos) - 1):
        ch, br = compare(lasts[i], firsts[i + 1])
        bright_bad |= br > 8
        print(f"  join {i + 1} -> {i + 2}: {ch:.2f}% changed, brightness diff {br:.1f}  [{verdict(ch, limit)}]")
    loop_ch, loop_br = compare(lasts[-1], firsts[0])
    if not a.pingpong:
        bright_bad |= loop_br > 8
        print(f"  loop end -> start: {loop_ch:.2f}% changed, brightness diff {loop_br:.1f}  [{verdict(loop_ch, limit)}]")
    if bright_bad:
        print("  note: a brightness difference above 8 means the lighting differs between clips - regenerate that clip.")

    if a.pingpong:
        cap = cv2.VideoCapture(str(tmp))
        cyc = cv2.VideoWriter(str(tmp.with_suffix(".pp.avi")), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        for i in range(n):                                  # forward part
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if not ok:
                break
            cyc.write(fr)
        for i in range(n - 2, 0, -1):                       # backward part, no repeated end frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, fr = cap.read()
            if ok:
                cyc.write(fr)
        cap.release()
        cyc.release()
        tmp.unlink(missing_ok=True)
        tmp = tmp.with_suffix(".pp.avi")
        print("\nPing-pong added: the video now plays forward and then backward, so the end equals the start.")
    elif loop_ch > limit:
        print("\nThe end does not match the start, so every repeat will jump at the loop point."
              "\nGenerate the last clip with clip 1's first frame as its end image, or add --pingpong.")

    # ---- repeat and write ---------------------------------------------------------------------------
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*a.codec), fps, size)
    if not writer.isOpened():
        tmp.unlink(missing_ok=True)
        sys.exit(f"Cannot create {out_path} (codec {a.codec}). Try --codec avc1, or an .avi name with --codec XVID.")
    written = 0
    for r in range(1, a.repeat + 1):
        cap = cv2.VideoCapture(str(tmp))
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            writer.write(fr)
            written += 1
        cap.release()
        print(f"  round {r}/{a.repeat} written ({written / fps:.0f} s)")
    writer.release()
    tmp.unlink(missing_ok=True)
    print(f"\nSaved {out_path}: {written} frames, {written / fps:.1f} s ({written / fps / 60:.1f} min); "
          f"one round = {written / a.repeat / fps:.1f} s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("frames", help="save the first and last frame of each clip as PNG")
    f.add_argument("videos", nargs="+")
    f.add_argument("--out-dir", default="frames")
    f.set_defaults(func=cmd_frames)

    b = sub.add_parser("build", help="join clips, check joins and loop, repeat")
    b.add_argument("videos", nargs="+", help="clips in the order they should play")
    b.add_argument("--repeat", type=int, default=1, help="how many times to repeat the joined video (default 1)")
    b.add_argument("--out", default="long_test.mp4")
    b.add_argument("--size", default="1280x720", help="output size (default 1280x720, the size your ROI is made for)")
    b.add_argument("--fps", type=float, default=15.0, help="output frame rate (default 15, like your config)")
    b.add_argument("--pingpong", action="store_true", help="play forward then backward so the end equals the start")
    b.add_argument("--codec", default="mp4v")
    b.set_defaults(func=cmd_build)

    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()