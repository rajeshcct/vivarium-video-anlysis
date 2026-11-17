"""Mouse detection: find the mouse blob, track its centre, measure speed.

Tail-rejection strategy
-----------------------
A mouse body is roughly compact (low elongation, decent area).  The tail is a
thin, elongated blob.  We filter candidate contours by THREE shape tests before
accepting one as the mouse:

1. min_area      – as before (removes dust and small noise)
2. max_area      – blobs much larger than a mouse are cage artefacts or shadows
3. solidity      – contourArea / convexHullArea; a mouse body ≥ 0.4,
                   a tail ≤ 0.2 (thin rod has very convex hull).
4. aspect_ratio  – bounding-rectangle W/H; tail ≫ body.  We reject blobs
                   whose aspect_ratio > max_aspect (default 4.0).
5. proximity     – when multiple valid blobs pass the shape tests, prefer the
                   one closest to the last known position (not the largest).

All thresholds live in config.yaml under detection so they are tunable without
touching the code.
"""
import collections
import math

import cv2
import numpy as np


def _shape_ok(cnt, min_area, max_area, min_solidity, max_aspect):
    """Return True if contour looks like a mouse body, not a tail or shadow."""
    area = cv2.contourArea(cnt)
    if area < min_area:
        return False
    if max_area and area > max_area:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(cnt))
    if hull_area == 0:
        return False
    solidity = area / hull_area
    if solidity < min_solidity:
        return False
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > max_aspect:
        return False
    return True


class MotionDetector:
    def __init__(self, cfg, frame_shape):
        d, v, m = cfg["detection"], cfg["video"], cfg["movement"]
        h, w = frame_shape[:2]
        self.method      = d["method"]
        self.min_area    = d["min_area"]
        self.max_area    = d.get("max_area", None)        # e.g. 4 × min_area
        self.min_solidity = d.get("min_solidity", 0.35)  # mouse body ≥ 0.35
        self.max_aspect  = d.get("max_aspect", 4.0)      # tail aspect >> 4
        self.blur        = d["blur"]
        self.diff_thr    = d["diff_threshold"]
        self.lr          = d["learning_rate"]
        self.warmup      = d["warmup_sec"]
        self.speed_thr   = m["speed_threshold"]
        self.window      = m["speed_window_sec"]

        # Extra morphology: CLOSE gaps between body + tail before finding cnts
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (d.get("close_kernel", 15), d.get("close_kernel", 15)))

        self.roi = self._resolve_roi(v["roi"], w, h)
        _, _, rw, rh = self.roi
        pw = v["process_width"]
        self.size = (rw, rh) if pw is None else (pw, max(1, round(rh * pw / rw)))

        k = d["morph_kernel"]
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.t0        = None
        self.prev_gray = None
        self.track     = collections.deque()
        self.last_pos  = None

        self.bg      = None
        self.bg_gray = None
        if self.method == "mog2":
            self.bg = cv2.createBackgroundSubtractorMOG2(
                history=d["history"], varThreshold=d["var_threshold"], detectShadows=False)
        elif self.method == "fixed_background":
            img = cv2.imread(str(d["background_image"]))
            if img is None:
                raise RuntimeError(f"Cannot read background image: {d['background_image']}")
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            _, self.bg_gray = self._prep(img)

    @staticmethod
    def _resolve_roi(roi, w, h):
        if roi is None:
            return (0, 0, w, h)
        x, y, rw, rh = roi
        x, y = min(x, w - 1), min(y, h - 1)
        return (x, y, min(rw, w - x), min(rh, h - y))

    def _prep(self, frame):
        x, y, rw, rh = self.roi
        crop  = frame[y:y + rh, x:x + rw]
        small = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        if self.blur > 1:
            gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        return small, gray

    def _mask(self, gray):
        if self.method == "mog2":
            mask = self.bg.apply(gray, learningRate=self.lr)
        elif self.method == "fixed_background":
            mask = cv2.absdiff(gray, self.bg_gray)
        else:  # frame_diff
            mask = np.zeros_like(gray) if self.prev_gray is None else cv2.absdiff(gray, self.prev_gray)
            self.prev_gray = gray
        thr  = 127 if self.method == "mog2" else self.diff_thr
        mask = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self.kernel)
        # CLOSE: merge body + nearby tail blob into one before shape test
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        return mask

    def _best_contour(self, cnts):
        """
        From a list of contours that passed the area floor, pick the one that
        looks most like a mouse body:
          1. Must pass shape tests (solidity, aspect ratio, max area).
          2. Among passing candidates, prefer the one nearest the last known
             position rather than blindly taking the largest.
        """
        candidates = [c for c in cnts if _shape_ok(
            c, self.min_area, self.max_area, self.min_solidity, self.max_aspect)]

        if not candidates:
            # All blobs failed shape test — fall back to largest raw blob
            # (better than losing the mouse entirely)
            return max(cnts, key=cv2.contourArea)

        if self.last_pos is None or len(candidates) == 1:
            return max(candidates, key=cv2.contourArea)

        # Prefer blob whose centroid is closest to the last known position
        def _centroid(c):
            mo = cv2.moments(c)
            if mo["m00"] == 0:
                return (0.0, 0.0)
            return (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])

        lx, ly = self.last_pos
        return min(candidates,
                   key=lambda c: math.hypot(_centroid(c)[0] - lx, _centroid(c)[1] - ly))

    def process(self, frame, t):
        small, gray = self._prep(frame)
        if self.t0 is None:
            self.t0 = t
        mask    = self._mask(gray)
        warming = (t - self.t0) < self.warmup

        found, pos, area = False, None, 0.0
        if not warming:
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts = [c for c in cnts if cv2.contourArea(c) >= self.min_area]
            if cnts:
                c  = self._best_contour(cnts)
                mo = cv2.moments(c)
                if mo["m00"] > 0:
                    pos   = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
                    area  = cv2.contourArea(c)
                    found = True

        if found:
            self.last_pos = pos
            self.track.append((t, pos[0], pos[1]))
        else:
            pos = self.last_pos

        # speed = displacement over the last `window` seconds
        while self.track and t - self.track[0][0] > self.window:
            self.track.popleft()
        speed = 0.0
        if found and len(self.track) >= 2:
            t_a, x_a, y_a = self.track[0]
            t_b, x_b, y_b = self.track[-1]
            if t_b > t_a:
                speed = math.hypot(x_b - x_a, y_b - y_a) / (t_b - t_a)

        return {
            "t":      t,
            "x":      pos[0] if pos else None,
            "y":      pos[1] if pos else None,
            "found":  found,
            "area":   area,
            "speed":  speed,
            "moving": (not warming) and speed > self.speed_thr,
            "small":  small,
            "mask":   mask,   # exposed for debug overlay
        }


def draw_overlay(res):
    img = res["small"].copy()
    moving = res["moving"]
    color = (0, 200, 0) if moving else (0, 0, 255)
    if res["x"] is not None:
        cv2.circle(img, (int(res["x"]), int(res["y"])), 5, color, -1)
        if not res["found"]:
            cv2.circle(img, (int(res["x"]), int(res["y"])), 9, (0, 200, 255), 1)   # last known position
    cv2.putText(img, "MOVING" if moving else "STILL", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"t={res['t']:.1f}s v={res['speed']:.1f}px/s", (8, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_overlay(res, show_mask=False):
    img    = res["small"].copy()
    moving = res["moving"]
    color  = (0, 200, 0) if moving else (0, 0, 255)
    if res["x"] is not None:
        cv2.circle(img, (int(res["x"]), int(res["y"])), 5, color, -1)
        if not res["found"]:
            cv2.circle(img, (int(res["x"]), int(res["y"])), 9, (0, 200, 255), 1)  # last known
    cv2.putText(img, "MOVING" if moving else "STILL", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"t={res['t']:.1f}s v={res['speed']:.1f}px/s a={res['area']:.0f}px2",
                (8, img.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    if show_mask and "mask" in res:
        # Side-by-side: overlay | mask (useful for tuning)
        mask_bgr = cv2.cvtColor(res["mask"], cv2.COLOR_GRAY2BGR)
        img = np.hstack([img, mask_bgr])
    return img
