"""Movement trail on saved clips, plus line graphs. Used by main.py (auto) and tools/render_trail.py (manual).

Original clips are never changed: trail videos go to <session>/trails/.
"""
import bisect
import csv
import logging
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


def load_frames(path):
    t, x, y, found, moving, speed = [], [], [], [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t.append(float(r["t_s"]))
            x.append(float(r["x_px"]) if r["x_px"] != "" else None)
            y.append(float(r["y_px"]) if r["y_px"] != "" else None)
            found.append(r["found"] == "1")
            moving.append(r["moving"] == "1")
            speed.append(float(r["speed_px_s"]))
    return t, x, y, found, moving, speed


def make_mapper(cfg, W, H):
    """Process-size coordinates (inside the ROI) -> full clip pixel coordinates."""
    roi = cfg["video"]["roi"]
    if roi is None:
        rx, ry, rw, rh = 0, 0, W, H
    else:
        x, y, rw, rh = roi
        rx, ry = min(x, W - 1), min(y, H - 1)
        rw, rh = min(rw, W - rx), min(rh, H - ry)
    pw = cfg["video"]["process_width"]
    if pw is None:
        pw, ph = rw, rh
    else:
        ph = max(1, round(rh * pw / rw))
    sx, sy = rw / pw, rh / ph
    return lambda px, py: (int(rx + px * sx), int(ry + py * sy))


def render_clip(src, dst, start_s, data, cfg, trail_sec, thickness):
    t, x, y, found, moving, _ = data
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        log.warning("Trail: cannot open %s, skipped", src)
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    mapper = make_mapper(cfg, W, H)
    out = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*cfg["recording"]["codec"]), fps, (W, H))
    if not out.isOpened():
        log.warning("Trail: cannot create %s, skipped", dst)
        cap.release()
        return False

    # only frames where the mouse was really detected are used for the line
    pts = [(t[k], *mapper(x[k], y[k])) for k in range(len(t)) if found[k] and x[k] is not None]
    ptimes = [p[0] for p in pts]

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ti = start_s + i / fps
        i += 1
        hi = bisect.bisect_right(ptimes, ti)
        lo = bisect.bisect_left(ptimes, ti - trail_sec) if trail_sec > 0 else bisect.bisect_left(ptimes, start_s)
        seg = pts[lo:hi]
        n = len(seg)
        for k in range(1, n):
            a = k / n                                           # 0 = oldest, 1 = newest
            color = (int(255 * (1 - a)), 80, int(255 * a))      # blue (old) -> red (new)
            cv2.line(frame, seg[k - 1][1:], seg[k][1:], color, thickness, cv2.LINE_AA)
        if n:
            cv2.circle(frame, seg[-1][1:], thickness + 3, (0, 255, 255), -1, cv2.LINE_AA)

        idx = bisect.bisect_right(t, ti) - 1
        is_moving = idx >= 0 and moving[idx]
        label_color = (0, 200, 0) if is_moving else (0, 0, 255)
        cv2.putText(frame, "MOVING" if is_moving else "STILL", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, label_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"t={ti:.1f}s", (12, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        out.write(frame)
    out.release()
    cap.release()
    return True


def make_graphs(session_dir, cfg, data):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
    except ImportError:
        log.warning("matplotlib is not installed, graphs skipped "
                    "(laptop: pip install matplotlib | Pi: sudo apt install python3-matplotlib)")
        return

    session_dir = Path(session_dir)
    t, x, y, found, _, speed = data
    ppc = cfg["movement"]["px_per_cm"]
    unit, k = ("cm", 1.0 / ppc) if ppc else ("px", 1.0)

    P = [(t[i], x[i] * k, y[i] * k) for i in range(len(t)) if found[i] and x[i] is not None]
    if len(P) >= 2:
        pts = [(p[1], p[2]) for p in P]
        segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
        lc = LineCollection(segs, cmap="viridis", linewidths=1.5)
        lc.set_array([p[0] for p in P[:-1]])
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.add_collection(lc)
        ax.scatter(*pts[0], c="green", s=60, zorder=3, label="start")
        ax.scatter(*pts[-1], c="red", s=60, zorder=3, label="end")
        ax.autoscale()
        ax.set_aspect("equal")
        ax.invert_yaxis()
        ax.set_xlabel(f"x ({unit})")
        ax.set_ylabel(f"y ({unit})")
        ax.set_title("Movement path")
        ax.legend()
        fig.colorbar(lc, ax=ax, label="time (s)")
        fig.savefig(session_dir / "trajectory.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info("Saved %s", session_dir / "trajectory.png")
    else:
        log.warning("Not enough detected positions for the trajectory graph")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, [s * k for s in speed], lw=0.8, color="tab:blue", label="speed")
    ax.axhline(cfg["movement"]["speed_threshold"] * k, color="red", ls="--", lw=1, label="speed threshold")
    tl = session_dir / "timeline.csv"
    if tl.is_file():
        with open(tl, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["state"] == "moving":
                    ax.axvspan(float(r["start_s"]), float(r["end_s"]), color="green", alpha=0.18, lw=0)
        ax.axvspan(0, 0, color="green", alpha=0.18, label="moving (timeline)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"speed ({unit}/s)")
    ax.set_title("Speed over time")
    ax.set_xlim(left=0)
    ax.legend(loc="upper right")
    fig.savefig(session_dir / "speed_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", session_dir / "speed_timeline.png")


def render_session(session_dir, cfg, trail_sec=5.0, thickness=2, clips=True, graphs=True):
    """Make trail clips and graphs for one finished session folder."""
    sd = Path(session_dir)
    fd = sd / "frame_data.csv"
    if not fd.is_file():
        log.warning("Trail: frame_data.csv not found in %s (needs output.save_frame_data: true)", sd)
        return False
    data = load_frames(fd)
    if not data[0]:
        log.warning("Trail: frame_data.csv is empty")
        return False

    if clips:
        clips_csv = sd / "clips.csv"
        if not clips_csv.is_file():
            log.info("Trail: no clips.csv (recording was off), no clips to draw on")
        else:
            cdir = Path(cfg["recording"]["clip_dir"])
            cdir = cdir if cdir.is_absolute() else sd / cdir
            tdir = sd / "trails"
            with open(clips_csv, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                tdir.mkdir(exist_ok=True)
                log.info("Drawing movement trail on %d clip(s) -> %s", len(rows), tdir)
            else:
                log.info("Trail: no clips were recorded in this session")
            for r in rows:
                dst = tdir / (Path(r["clip"]).stem + "_trail.mp4")
                if render_clip(cdir / r["clip"], dst, float(r["start_s"]), data, cfg, trail_sec, thickness):
                    log.info("  saved %s", dst)
    if graphs:
        make_graphs(sd, cfg, data)
    return True
