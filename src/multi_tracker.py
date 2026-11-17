"""Multi-mouse tracking on top of the existing background-subtraction detector.

Pipeline per frame:
  1. foreground mask (same MOG2 / fixed_background / frame_diff code as before)
  2. keep the n_mice biggest blobs (smaller ones are noise)
  3. a blob much bigger than one mouse = touching mice -> split it with a small
     k-means seeded from the previous mouse positions (keeps IDs through contact)
  4. match detections to tracks by nearest distance (gated, no scipy needed)
  5. speed and moving/still per mouse

Set detection.n_mice to 2 or more to use it. With n_mice = 1 nothing changes.
"""
import collections
import itertools
import math

import cv2
import numpy as np

from src.detector import MotionDetector, draw_overlay as _draw_single

COLORS = [(0, 200, 0), (255, 120, 0), (0, 200, 255), (200, 0, 200), (255, 255, 0), (0, 128, 255)]  # BGR
MAX_KMEANS_POINTS = 3000


class _Track:
    def __init__(self, tid, pos, t):
        self.id = tid
        self.pos = pos
        self.found = True
        self.merged = False
        self.missed = 0
        self.hist = collections.deque([(t, pos[0], pos[1])])
        self.speed = 0.0


def _assign(cost, gates):
    """Match tracks (rows) to detections (cols). Most valid matches first, then smallest total distance."""
    nt, nd = cost.shape
    if nt == 0 or nd == 0:
        return []
    big = 1e6
    c = np.where(cost <= gates[:, None], cost, big)
    pairs = []
    if max(nt, nd) <= 6:
        best, best_sum = None, None
        if nt <= nd:
            for perm in itertools.permutations(range(nd), nt):
                s = sum(c[i, j] for i, j in enumerate(perm))
                if best_sum is None or s < best_sum:
                    best, best_sum = [(i, j) for i, j in enumerate(perm)], s
        else:
            for perm in itertools.permutations(range(nt), nd):
                s = sum(c[i, j] for j, i in enumerate(perm))
                if best_sum is None or s < best_sum:
                    best, best_sum = [(i, j) for j, i in enumerate(perm)], s
        pairs = best
    else:  # many animals: greedy
        order = sorted(((c[i, j], i, j) for i in range(nt) for j in range(nd)))
        used_i, used_j = set(), set()
        for v, i, j in order:
            if i not in used_i and j not in used_j:
                pairs.append((i, j))
                used_i.add(i)
                used_j.add(j)
    return [(i, j) for i, j in pairs if c[i, j] < big]


def _farthest_seeds(pts, k, have):
    seeds = list(have)
    if not seeds:
        seeds.append(pts[np.argmax(np.linalg.norm(pts - pts.mean(0), axis=1))])
    while len(seeds) < k:
        d = np.min([np.linalg.norm(pts - s, axis=1) for s in seeds], axis=0)
        seeds.append(pts[np.argmax(d)])
    return np.array(seeds, np.float32)


class MultiMouseDetector(MotionDetector):
    def __init__(self, cfg, frame_shape):
        super().__init__(cfg, frame_shape)
        d = cfg["detection"]
        self.n = d["n_mice"]
        self.split_factor = d["split_area_factor"]
        self.max_dist = d["max_match_dist"]
        self.ref_area = d["mouse_area"]            # None = learn from frames where every mouse is separate
        self._areas = collections.deque(maxlen=200)
        self.tracks = []
        self._area = 0.0

    # ---- detections -------------------------------------------------------------------------------
    def _split(self, contour, k, centre):
        x, y, w, h = cv2.boundingRect(contour)
        blob = np.zeros((h, w), np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, -1, offset=(-x, -y))
        ys, xs = np.nonzero(blob)
        pts = np.column_stack((xs + x, ys + y)).astype(np.float32)
        if len(pts) > MAX_KMEANS_POINTS:
            pts = pts[::math.ceil(len(pts) / MAX_KMEANS_POINTS)]
        near = sorted(self.tracks, key=lambda tr: math.hypot(tr.pos[0] - centre[0], tr.pos[1] - centre[1]))[:k]
        centers = _farthest_seeds(pts, k, [np.array(tr.pos, np.float32) for tr in near])
        for _ in range(10):
            dist = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
            lab = dist.argmin(1)
            new = centers.copy()
            for j in range(k):
                m = lab == j
                if m.any():
                    new[j] = pts[m].mean(0)
            if np.allclose(new, centers, atol=0.1):
                centers = new
                break
            centers = new
        return [(float(cx), float(cy)) for cx, cy in centers]

    def _detections(self, mask):
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        blobs = [(cv2.contourArea(c), c) for c in cnts]
        blobs = sorted((b for b in blobs if b[0] >= self.min_area), key=lambda b: -b[0])[:self.n]
        self._area = float(sum(b[0] for b in blobs))
        if not blobs:
            return []
        areas = [b[0] for b in blobs]
        if len(blobs) == self.n:                    # every mouse has its own blob: learn single-mouse size
            self._areas.append(float(np.median(areas)))
        ref = self.ref_area
        if ref is None and len(self._areas) >= 10:
            ref = float(np.median(self._areas))

        counts = [1] * len(blobs)
        if ref and len(blobs) < self.n:             # fewer blobs than mice: someone is touching
            for _ in range(self.n - len(blobs)):
                i = max(range(len(blobs)), key=lambda q: areas[q] / counts[q])
                if areas[i] >= ref * (counts[i] + self.split_factor - 1):
                    counts[i] += 1
                else:
                    break                           # the rest are probably not visible (e.g. motionless)

        dets = []
        for (a, c), k in zip(blobs, counts):
            mo = cv2.moments(c)
            if mo["m00"] <= 0:
                continue
            centre = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
            if k == 1:
                dets.append((centre[0], centre[1], False))
            else:
                dets.extend((px, py, True) for px, py in self._split(c, k, centre))
        return dets

    # ---- tracks -----------------------------------------------------------------------------------
    def _update_tracks(self, dets, t):
        matched_t, matched_d = set(), set()
        if self.tracks and dets:
            cost = np.array([[math.hypot(tr.pos[0] - dx, tr.pos[1] - dy) for dx, dy, _ in dets]
                             for tr in self.tracks])
            gates = np.array([self.max_dist * min(3.0, 1.0 + 0.5 * tr.missed) for tr in self.tracks])
            for i, j in _assign(cost, gates):
                tr, (dx, dy, merged) = self.tracks[i], dets[j]
                tr.pos, tr.found, tr.merged, tr.missed = (dx, dy), True, merged, 0
                tr.hist.append((t, dx, dy))
                matched_t.add(i)
                matched_d.add(j)
        for i, tr in enumerate(self.tracks):
            if i not in matched_t:
                tr.found, tr.merged = False, False
                tr.missed += 1
        for j, (dx, dy, merged) in sorted(((j, d) for j, d in enumerate(dets) if j not in matched_d),
                                          key=lambda q: (q[1][0], q[1][1])):
            if len(self.tracks) < self.n:
                tr = _Track(len(self.tracks) + 1, (dx, dy), t)
                tr.merged = merged
                self.tracks.append(tr)

    def process(self, frame, t):
        small, gray = self._prep(frame)
        if self.t0 is None:
            self.t0 = t
        mask = self._mask(gray)
        warming = (t - self.t0) < self.warmup
        dets = [] if warming else self._detections(mask)
        self._update_tracks(dets, t)

        mice = []
        for tr in self.tracks:
            while tr.hist and t - tr.hist[0][0] > self.window:
                tr.hist.popleft()
            tr.speed = 0.0
            if tr.found and len(tr.hist) >= 2:
                t_a, x_a, y_a = tr.hist[0]
                t_b, x_b, y_b = tr.hist[-1]
                if t_b > t_a:
                    tr.speed = math.hypot(x_b - x_a, y_b - y_a) / (t_b - t_a)
            mice.append({"id": tr.id, "x": tr.pos[0], "y": tr.pos[1], "found": tr.found,
                         "speed": tr.speed, "moving": (not warming) and tr.speed > self.speed_thr,
                         "merged": tr.merged})

        first = mice[0] if mice else None
        return {
            "t": t,
            # cage-level values keep the old output files working:
            "x": first["x"] if first else None,       # position of mouse 1
            "y": first["y"] if first else None,
            "found": len(mice) == self.n and all(m["found"] for m in mice),   # every mouse seen
            "area": self._area,
            "speed": max((m["speed"] for m in mice), default=0.0),            # fastest mouse
            "moving": any(m["moving"] for m in mice),                         # any mouse moving
            "small": small,
            "mice": mice,
        }


def make_detector(cfg, frame_shape):
    if cfg["detection"]["n_mice"] > 1:
        return MultiMouseDetector(cfg, frame_shape)
    return MotionDetector(cfg, frame_shape)


def draw_overlay(res):
    mice = res.get("mice")
    if not mice:
        return _draw_single(res)
    img = res["small"].copy()
    for m in mice:
        if m["x"] is None:
            continue
        col = COLORS[(m["id"] - 1) % len(COLORS)]
        p = (int(m["x"]), int(m["y"]))
        cv2.circle(img, p, 5, col, -1)
        if not m["found"]:
            cv2.circle(img, p, 9, (0, 200, 255), 1)           # last known position
        elif m["merged"]:
            cv2.circle(img, p, 9, (255, 255, 255), 1)         # position from a split blob
        cv2.putText(img, f"M{m['id']}{' moving' if m['moving'] else ''}", (p[0] + 8, p[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
    txt = "ANY MOVING" if res["moving"] else "ALL STILL"
    cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 200, 0) if res["moving"] else (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(img, f"t={res['t']:.1f}s", (8, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img
