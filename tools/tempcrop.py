"""
tempcrop.py  –  Trim a video to a given start and end time.

Usage:
    python tools/tempcrop.py <input_video> <start_time> <end_time> [output_file]

Times can be given as:
    seconds          : 30 or 90.5
    MM:SS            : 1:30
    HH:MM:SS         : 0:01:30

If output_file is omitted, the result is saved next to the input video
with "_crop" appended to the filename.

Example:
    python tools/tempcrop.py samples/test_mouse.mp4 10 45
    python tools/tempcrop.py samples/test_mouse.mp4 1:30 3:00 output/clip.mp4
"""

import sys
import os
import cv2


def parse_time(s):
    """Convert HH:MM:SS, MM:SS, or plain seconds string to a float (seconds)."""
    parts = s.strip().split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            raise ValueError
    except ValueError:
        print(f"ERROR: Cannot parse time '{s}'. Use seconds (30.5) or MM:SS or HH:MM:SS.")
        sys.exit(1)


def crop_video(input_path, start_sec, end_sec, output_path):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {input_path}")
        sys.exit(1)

    fps       = cap.get(cv2.CAP_PROP_FPS)
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_sec = total_frames / fps if fps else 0

    print(f"Source      : {input_path}")
    print(f"Resolution  : {width}x{height}  |  FPS: {fps:.2f}  |  Duration: {total_sec:.2f}s")

    if start_sec < 0:
        start_sec = 0.0
    if end_sec <= 0 or end_sec > total_sec:
        end_sec = total_sec
    if start_sec >= end_sec:
        print(f"ERROR: start ({start_sec}s) must be before end ({end_sec}s).")
        sys.exit(1)

    start_frame = int(start_sec * fps)
    end_frame   = int(end_sec   * fps)
    duration    = end_sec - start_sec
    print(f"Cropping    : {start_sec:.2f}s → {end_sec:.2f}s  ({duration:.2f}s, ~{end_frame - start_frame} frames)")
    print(f"Output      : {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0
    for frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            print(f"  Warning: stream ended early at frame {frame_idx}")
            break
        out.write(frame)
        written += 1
        if written % int(fps * 5) == 0:          # progress every ~5 s
            elapsed = written / fps
            print(f"  {elapsed:.1f}s / {duration:.1f}s written…")

    cap.release()
    out.release()
    print(f"Done. {written} frames written to: {output_path}")


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        sys.exit(0)

    input_path = sys.argv[1]
    start_sec  = parse_time(sys.argv[2])
    end_sec    = parse_time(sys.argv[3])

    if len(sys.argv) >= 5:
        output_path = sys.argv[4]
    else:
        base, ext = os.path.splitext(input_path)
        output_path = base + "_crop" + (ext if ext else ".mp4")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    crop_video(input_path, start_sec, end_sec, output_path)


if __name__ == "__main__":
    main()

