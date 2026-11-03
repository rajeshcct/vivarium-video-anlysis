"""Mouse detection: find the mouse blob, track its centre, measure speed."""
import collections
import math

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, cfg, frame_shape):
        d, v, m = cfg["detection"], cfg["video"], cfg["movement"]
        h, w = frame_shape[:2]
        self.method = d["method"]
        self.min_area = d["min_area"]
        self.blur = d["blur"]
        self.diff_thr = d["diff_threshold"]
        self.lr = d["learning_rate"]
        self.warmup = d["warmup_sec"]
        self.speed_thr = m["speed_threshold"]
        self.window = m["speed_window_sec"]

        self.roi = self._resolve_roi(v["roi"], w, h)
        _, _, rw, rh = self.roi
        pw = v["process_width"]
        self.size = (rw, rh) if pw is None else (pw, max(1, round(rh * pw / rw)))

        k = d["morph_kernel"]
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.t0 = None
        self.prev_gray = None
        self.track = collections.deque()
        self.last_pos = None

        self.bg = None
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
        crop = frame[y:y + rh, x:x + rw]
        small = cv2.resize(crop, self.size, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
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
        thr = 127 if self.method == "mog2" else self.diff_thr
        mask = cv2.threshold(mask, thr, 255, cv2.THRESH_BINARY)[1]
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

    def process(self, frame, t):
        small, gray = self._prep(frame)
        if self.t0 is None:
            self.t0 = t
        mask = self._mask(gray)
        warming = (t - self.t0) < self.warmup

        found, pos, area = False, None, 0.0
        if not warming:
            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            cnts = [c for c in cnts if cv2.contourArea(c) >= self.min_area]
            if cnts:
                c = max(cnts, key=cv2.contourArea)
                mo = cv2.moments(c)
                if mo["m00"] > 0:
                    pos = (mo["m10"] / mo["m00"], mo["m01"] / mo["m00"])
                    area = cv2.contourArea(c)
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
            "t": t,
            "x": pos[0] if pos else None,
            "y": pos[1] if pos else None,
            "found": found,
            "area": area,
            "speed": speed,
            "moving": (not warming) and speed > self.speed_thr,
            "small": small,
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
