"""Movement graphs for one finished session. Used by main.py (auto) and tools/make_graphs.py (manual).

Reads frame_data.csv and timeline.csv from the session folder and writes PNG files next to them.
No video is read or written, so it is fast even for long sessions.

  speed_timeline.png       speed against time, speed threshold, moving periods shaded
  activity_bar.png         one bar across the session: moving / still
  moving_per_minute.png    % of each minute spent moving
  distance_cumulative.png  distance travelled so far
  trajectory.png           the path in the cage, coloured by time
  heatmap.png              where the mouse spent time
"""
import csv
import logging
import math
import statistics
from pathlib import Path

log = logging.getLogger(__name__)

BLUE = "#2a78d6"         # moving / main data
BLUE_LIGHT = "#bcd6f5"
BLUE_DARK = "#0d3b78"
GREY = "#c9c8c2"         # still
INK = "#0b0b0b"
INK2 = "#52514e"
AXIS = "#9a9993"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


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


def load_runs(path):
    runs = []
    if path.is_file():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                runs.append((float(r["start_s"]), float(r["end_s"]), r["state"]))
    return runs


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.edgecolor": AXIS, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "font.size": 10,
    })
    return plt


def _clean(ax, grid="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)


def _save(plt, fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s", path)


def _time_scale(end_s):
    """Seconds for short sessions, minutes for long ones."""
    return (60.0, "min") if end_s >= 180 else (1.0, "s")


def _unit(cfg):
    ppc = cfg["movement"]["px_per_cm"]
    return ("cm", 1.0 / ppc) if ppc else ("px", 1.0)


def _pct_moving(runs):
    total = runs[-1][1] - runs[0][0]
    mov = sum(e - s for s, e, st in runs if st == "moving")
    return 100.0 * mov / total if total > 0 else 0.0


# ---------------------------------------------------------------- graphs

def graph_speed(plt, sd, cfg, data, runs):
    from matplotlib.patches import Patch
    t, _, _, _, _, speed = data
    unit, k = _unit(cfg)
    div, tu = _time_scale(t[-1])
    fig, ax = plt.subplots(figsize=(11, 4))
    for s, e, st in runs:
        if st == "moving":
            ax.axvspan(s / div, e / div, color=BLUE, alpha=0.12, lw=0)
    ax.plot([v / div for v in t], [s * k for s in speed], lw=1.4, color=BLUE)
    thr = cfg["movement"]["speed_threshold"] * k
    ax.axhline(thr, color=INK2, ls="--", lw=1)
    ax.set_xlabel(f"time ({tu})")
    ax.set_ylabel(f"speed ({unit}/s)")
    ax.set_title("Speed over time")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _clean(ax)
    handles = [plt.Line2D([], [], color=BLUE, lw=1.6, label="speed"),
               plt.Line2D([], [], color=INK2, ls="--", lw=1, label="speed threshold")]
    if runs:
        handles.append(Patch(color=BLUE, alpha=0.12, label="moving period"))
    ax.legend(handles=handles, loc="upper right", frameon=False)
    _save(plt, fig, sd / "speed_timeline.png")


def graph_activity_bar(plt, sd, cfg, data, runs):
    from matplotlib.patches import Patch
    if not runs:
        log.warning("Graph activity_bar skipped: timeline.csv not found")
        return
    div, tu = _time_scale(runs[-1][1])
    fig, ax = plt.subplots(figsize=(11, 1.9))
    for s, e, st in runs:
        ax.broken_barh([(s / div, (e - s) / div)], (0, 1), facecolors=BLUE if st == "moving" else GREY,
                       edgecolor=SURFACE, linewidth=1.0)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, runs[-1][1] / div)
    ax.set_yticks([])
    ax.set_xlabel(f"time ({tu})")
    ax.set_title(f"Moving {_pct_moving(runs):.0f}% of the session")
    _clean(ax, grid=None)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(color=BLUE, label="moving"), Patch(color=GREY, label="still")],
              loc="upper left", bbox_to_anchor=(0, -0.5), ncol=2, frameon=False)
    _save(plt, fig, sd / "activity_bar.png")


def graph_per_minute(plt, sd, cfg, data, runs):
    if not runs:
        log.warning("Graph moving_per_minute skipped: timeline.csv not found")
        return
    end = runs[-1][1]
    n = int(math.ceil(end / 60.0))
    centres, pcts = [], []
    for b in range(n):
        lo, hi = b * 60.0, min((b + 1) * 60.0, end)
        if hi - lo <= 0:
            continue
        mov = sum(max(0.0, min(e, hi) - max(s, lo)) for s, e, st in runs if st == "moving")
        centres.append((lo + hi) / 2 / 60.0)
        pcts.append(100.0 * mov / (hi - lo))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(centres, pcts, width=0.82, color=BLUE)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(1.0, end / 60.0))
    ax.set_xlabel("time (min)")
    ax.set_ylabel("time moving (%)")
    ax.set_title("Movement per minute")
    _clean(ax)
    _save(plt, fig, sd / "moving_per_minute.png")


def graph_distance(plt, sd, cfg, data, runs):
    t, x, y, found, _, _ = data
    unit, k = _unit(cfg)
    jitter = cfg["movement"]["jitter_px"]
    div, tu = _time_scale(t[-1])
    total, ref, cum = 0.0, None, []
    for xi, yi, fi in zip(x, y, found):          # same rule as summary.csv (steps below jitter_px ignored)
        if fi and xi is not None:
            if ref is None:
                ref = (xi, yi)
            else:
                d = math.hypot(xi - ref[0], yi - ref[1])
                if d >= jitter:
                    total += d
                    ref = (xi, yi)
        cum.append(total * k)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot([v / div for v in t], cum, lw=1.6, color=BLUE)
    ax.annotate(f"{cum[-1]:.0f} {unit}", (t[-1] / div, cum[-1]), xytext=(-6, 8),
                textcoords="offset points", ha="right", color=INK2)
    ax.set_xlabel(f"time ({tu})")
    ax.set_ylabel(f"distance ({unit})")
    ax.set_title("Distance travelled")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    _clean(ax)
    _save(plt, fig, sd / "distance_cumulative.png")


def graph_trajectory(plt, sd, cfg, data, runs):
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    t, x, y, found, _, _ = data
    unit, k = _unit(cfg)
    P = [(t[i], x[i] * k, y[i] * k) for i in range(len(t)) if found[i] and x[i] is not None]
    if len(P) < 2:
        log.warning("Graph trajectory skipped: not enough detected positions")
        return
    div, tu = _time_scale(t[-1])
    pts = [(p[1], p[2]) for p in P]
    segs = [[pts[i], pts[i + 1]] for i in range(len(pts) - 1)]
    cmap = LinearSegmentedColormap.from_list("t", [BLUE_LIGHT, BLUE_DARK])
    lc = LineCollection(segs, cmap=cmap, linewidths=1.6)
    lc.set_array([p[0] / div for p in P[:-1]])
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.add_collection(lc)
    ax.scatter(*pts[0], s=70, marker="o", facecolor=SURFACE, edgecolor=INK, linewidth=1.6, zorder=3, label="start")
    ax.scatter(*pts[-1], s=70, marker="s", facecolor=INK, edgecolor=INK, zorder=3, label="end")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title("Movement path")
    _clean(ax, grid=None)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    cb = fig.colorbar(lc, ax=ax, label=f"time ({tu})")
    cb.outline.set_visible(False)
    _save(plt, fig, sd / "trajectory.png")


def graph_heatmap(plt, sd, cfg, data, runs):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    t, x, y, found, _, _ = data
    unit, k = _unit(cfg)
    idx = [i for i in range(len(t)) if found[i] and x[i] is not None]
    if len(idx) < 2:
        log.warning("Graph heatmap skipped: not enough detected positions")
        return
    xs = np.array([x[i] * k for i in idx])
    ys = np.array([y[i] * k for i in idx])
    xr, yr = xs.max() - xs.min(), ys.max() - ys.min()
    if xr <= 0 or yr <= 0:
        log.warning("Graph heatmap skipped: the mouse did not move")
        return
    dts = [b - a for a, b in zip(t, t[1:]) if b > a]
    dt = statistics.median(dts) if dts else 0.0
    nx = 40
    ny = max(5, int(round(nx * yr / xr)))
    counts, xe, ye = np.histogram2d(xs, ys, bins=(nx, ny))
    secs = np.ma.masked_equal(counts.T * dt, 0)          # rows = y, columns = x; empty cells stay blank
    cmap = LinearSegmentedColormap.from_list("h", [BLUE_LIGHT, BLUE_DARK])   # lightest visited cell still shows
    cmap.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(7, 6))
    mesh = ax.pcolormesh(xe, ye, secs, cmap=cmap)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title("Where the mouse spent time")
    _clean(ax, grid=None)
    cb = fig.colorbar(mesh, ax=ax, label="time in cell (s)")
    cb.outline.set_visible(False)
    _save(plt, fig, sd / "heatmap.png")


GRAPHS = (graph_speed, graph_activity_bar, graph_per_minute, graph_distance, graph_trajectory, graph_heatmap)


def render_graphs(session_dir, cfg):
    """Make all graphs for one finished session folder. Returns False if nothing could be made."""
    sd = Path(session_dir)
    fd = sd / "frame_data.csv"
    if not fd.is_file():
        log.warning("Graphs: frame_data.csv not found in %s (needs output.save_frame_data: true)", sd)
        return False
    data = load_frames(fd)
    if not data[0]:
        log.warning("Graphs: frame_data.csv is empty")
        return False
    try:
        plt = _setup()
    except ImportError:
        log.warning("matplotlib is not installed, graphs skipped "
                    "(laptop: pip install matplotlib | Pi: sudo apt install python3-matplotlib)")
        return False
    runs = load_runs(sd / "timeline.csv")
    log.info("Making graphs in %s", sd)
    for fn in GRAPHS:
        try:
            fn(plt, sd, cfg, data, runs)
        except Exception as e:                       # one failing graph must not stop the others
            plt.close("all")
            log.warning("%s failed: %s", fn.__name__, e)
    return True
