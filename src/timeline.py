"""Turn per-frame moving/still flags into bouts and summary measures."""
import math
import statistics


def frames_to_runs(times, moving):
    """Per-frame flags -> list of [start, end, state] runs."""
    if not times:
        return []
    dts = [b - a for a, b in zip(times, times[1:]) if b > a]
    end = times[-1] + (statistics.median(dts) if dts else 0.0)
    runs, start, cur = [], times[0], moving[0]
    for t, m in zip(times[1:], moving[1:]):
        if m != cur:
            runs.append([start, t, "moving" if cur else "still"])
            start, cur = t, m
    runs.append([start, end, "moving" if cur else "still"])
    return runs


def merge_short(runs, min_move, min_still):
    """Absorb runs that are shorter than their minimum into the neighbouring run.

    A moving run shorter than min_move becomes still (small twitch ignored).
    A still run shorter than min_still becomes moving (short pause ignored).
    """
    limits = {"moving": min_move, "still": min_still}
    runs = [list(r) for r in runs]
    while len(runs) > 1:
        bad = [k for k, r in enumerate(runs) if r[1] - r[0] < limits[r[2]]]
        if not bad:
            break
        i = min(bad, key=lambda k: runs[k][1] - runs[k][0])
        if i == 0:
            j = 1
        elif i == len(runs) - 1:
            j = i - 1
        else:
            left = runs[i - 1][1] - runs[i - 1][0]
            right = runs[i + 1][1] - runs[i + 1][0]
            j = i - 1 if left >= right else i + 1
        runs[j][0] = min(runs[j][0], runs[i][0])
        runs[j][1] = max(runs[j][1], runs[i][1])
        del runs[i]
        merged = [runs[0]]
        for r in runs[1:]:
            if r[2] == merged[-1][2]:
                merged[-1][1] = r[1]
            else:
                merged.append(r)
        runs = merged
    return runs


def _distance(rows, jitter_px):
    """Path length, ignoring steps smaller than jitter_px (removes centroid noise)."""
    total, ref = 0.0, None
    for _, x, y, found, _, _ in rows:
        if not found or x is None:
            continue
        if ref is None:
            ref = (x, y)
            continue
        d = math.hypot(x - ref[0], y - ref[1])
        if d >= jitter_px:
            total += d
            ref = (x, y)
    return total


def summarize(rows, runs, cfg, meta):
    """rows: (t, x, y, found, speed, moving) per frame."""
    session_sec = runs[-1][1] - runs[0][0] if runs else 0.0
    moving_sec = sum(e - s for s, e, st in runs if st == "moving")
    still_sec = sum(e - s for s, e, st in runs if st == "still")
    bouts = sum(1 for r in runs if r[2] == "moving")
    longest_still = max([e - s for s, e, st in runs if st == "still"], default=0.0)
    longest_moving = max([e - s for s, e, st in runs if st == "moving"], default=0.0)
    found_pct = 100.0 * sum(1 for r in rows if r[3]) / len(rows) if rows else 0.0

    dist_px = _distance(rows, cfg["movement"]["jitter_px"])
    out = dict(meta)
    out.update({
        "session_sec": round(session_sec, 2),
        "moving_sec": round(moving_sec, 2),
        "still_sec": round(still_sec, 2),
        "pct_moving": round(100.0 * moving_sec / session_sec, 1) if session_sec else 0.0,
        "moving_bouts": bouts,
        "longest_moving_sec": round(longest_moving, 2),
        "longest_still_sec": round(longest_still, 2),
        "distance_px": round(dist_px, 1),
        "mean_speed_px_s": round(dist_px / session_sec, 2) if session_sec else 0.0,
        "mouse_found_pct": round(found_pct, 1),
    })
    ppc = cfg["movement"]["px_per_cm"]
    if ppc:
        out["distance_cm"] = round(dist_px / ppc, 1)
        out["mean_speed_cm_s"] = round(dist_px / ppc / session_sec, 3) if session_sec else 0.0
    out.update({
        "fps": round(meta.get("fps", 0), 2),
        "speed_threshold": cfg["movement"]["speed_threshold"],
        "min_move_sec": cfg["movement"]["min_move_sec"],
        "min_still_sec": cfg["movement"]["min_still_sec"],
        "detection_method": cfg["detection"]["method"],
    })
    return out
