"""Clip recorder: keeps a pre-buffer in memory, saves a clip when movement starts,
and closes it after the mouse has been still for post_buffer_sec."""
import collections
import csv
import logging
from pathlib import Path

import cv2

log = logging.getLogger(__name__)


class ClipRecorder:
    def __init__(self, cfg, fps, frame_size, session_dir):
        r = cfg["recording"]
        self.fps = fps
        self.size = frame_size                       # (width, height) of the frames passed in
        self.continuous = r["continuous"]
        self.post = r["post_buffer_sec"]
        self.min_move = cfg["movement"]["min_move_sec"]
        self.move_since = None
        self.fourcc = cv2.VideoWriter_fourcc(*r["codec"])
        self.session_id = cfg["session"]["id"]

        clip_dir = Path(r["clip_dir"])
        self.dir = clip_dir if clip_dir.is_absolute() else Path(session_dir) / clip_dir
        self.dir.mkdir(parents=True, exist_ok=True)

        # pre-buffer must also cover the min_move_sec wait, so the clip starts before the movement did
        self.pre = collections.deque(maxlen=max(1, int((r["pre_buffer_sec"] + self.min_move) * fps)))
        self.writer = None
        self.n = 0
        self.start_t = 0.0
        self.last_motion_t = 0.0
        self.name = ""

        self._f = open(Path(session_dir) / "clips.csv", "w", newline="", encoding="utf-8")
        self._csv = csv.writer(self._f)
        self._csv.writerow(["clip", "start_s", "end_s", "duration_s"])

    def _open(self, start_t):
        self.n += 1
        self.name = f"{self.session_id}_clip_{self.n:03d}.mp4"
        self.writer = cv2.VideoWriter(str(self.dir / self.name), self.fourcc, self.fps, self.size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create clip file in {self.dir} (check recording.codec)")
        self.start_t = start_t

    def _close(self, end_t):
        if self.writer is None:
            return
        self.writer.release()
        self.writer = None
        self._csv.writerow([self.name, round(self.start_t, 2), round(end_t, 2), round(end_t - self.start_t, 2)])
        self._f.flush()

    def update(self, frame, t, moving):
        if self.continuous:
            if self.writer is None:
                self._open(t)
            self.writer.write(frame)
            return

        if self.writer is None:
            self.pre.append((frame, t))
            if not moving:
                self.move_since = None
            else:
                if self.move_since is None:
                    self.move_since = t
                if t - self.move_since >= self.min_move:     # sustained movement -> start clip
                    self._open(self.pre[0][1])
                    for f, _ in self.pre:
                        self.writer.write(f)
                    self.pre.clear()
                    self.last_motion_t = t
                    self.move_since = None
        else:
            self.writer.write(frame)
            if moving:
                self.last_motion_t = t
            elif t - self.last_motion_t >= self.post:
                self._close(t)

    def close(self, end_t):
        self._close(end_t)
        self._f.close()
