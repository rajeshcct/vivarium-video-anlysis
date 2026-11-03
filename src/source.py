"""Frame sources: video file, USB camera, Pi camera, RTSP stream.

Every source offers the same interface:
    source.read()    -> (ok, frame, t_seconds)
    source.fps       -> effective frames per second
    source.release()
The rest of the program never knows which input is used.
"""
import logging
import threading
import time

import cv2

from . import usbcam

log = logging.getLogger(__name__)

_ROTATE = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


class FrameSource:
    fps = 30.0
    rotate = 0

    def read(self):
        raise NotImplementedError

    def release(self):
        pass

    def _rotate(self, frame):
        code = _ROTATE[self.rotate]
        return frame if code is None else cv2.rotate(frame, code)


class FileSource(FrameSource):
    def __init__(self, cfg):
        s, v = cfg["source"], cfg["video"]
        self.rotate = v["rotate"]
        self.loop = s["loop"]
        self.cap = cv2.VideoCapture(str(s["path"]))
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {s['path']}")

        native = self.cap.get(cv2.CAP_PROP_FPS) or 0
        if native <= 1 or native > 1000:
            log.warning("Video reports FPS=%s, assuming 30", native)
            native = 30.0
        target = v["fps"]
        if target == "auto":
            target = native
        elif target > native:
            log.warning("video.fps=%s is higher than the file's %.2f fps; using %.2f", target, native, native)
            target = native
        self.native_fps = float(native)
        self.fps = float(target)
        self._ratio = self.native_fps / self.fps
        self._raw = 0
        self._next = 1.0
        self._out = 0

    def read(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self._raw, self._next = 0, 1.0
                    ok, frame = self.cap.read()
                if not ok:
                    return False, None, 0.0
            self._raw += 1
            if self._raw >= self._next:          # keep this frame
                self._next += self._ratio
                break
        t = self._out / self.fps
        self._out += 1
        return True, self._rotate(frame), t

    def release(self):
        self.cap.release()


class LiveSource(FrameSource):
    """Base for cameras and streams: wall-clock timestamps, FPS throttle, reconnect."""

    def __init__(self, cfg):
        self.cfg = cfg
        s, v = cfg["source"], cfg["video"]
        self.rotate = v["rotate"]
        self.target = None if v["fps"] == "auto" else float(v["fps"])
        self.attempts = s["reconnect_attempts"]
        self.wait = s["reconnect_wait_sec"]
        self._t0 = None
        self._last = None
        self._open()
        self.fps = self.target or self._native_fps() or 30.0

    # subclasses implement these
    def _open(self):
        raise NotImplementedError

    def _grab(self):
        raise NotImplementedError

    def _close(self):
        pass

    def _native_fps(self):
        return None

    def _recover(self):
        for i in range(1, self.attempts + 1):
            log.warning("Camera read failed, reconnect attempt %d/%d", i, self.attempts)
            try:
                self._close()
            except Exception:
                pass
            time.sleep(self.wait)
            try:
                self._open()
                ok, _ = self._grab()
                if ok:
                    log.info("Camera reconnected")
                    return True
            except Exception as e:
                log.warning("Reconnect failed: %s", e)
        return False

    def read(self):
        while True:
            ok, frame = self._grab()
            if not ok:
                if not self._recover():
                    return False, None, 0.0
                continue
            now = time.monotonic()
            if self._t0 is None:
                self._t0 = now
            t = now - self._t0
            if self.target and self._last is not None and (t - self._last) < 0.9 / self.target:
                continue                           # too soon, drop frame to match target fps
            self._last = t
            return True, self._rotate(frame), t

    def release(self):
        try:
            self._close()
        except Exception:
            pass


class _CvLiveSource(LiveSource):
    def _target(self):
        raise NotImplementedError

    def _setup(self, cap):
        pass

    def _open(self):
        self.cap = cv2.VideoCapture(self._target())
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera/stream: {self._target()}")
        self._setup(self.cap)

    def _grab(self):
        return self.cap.read()

    def _close(self):
        self.cap.release()

    def _native_fps(self):
        f = self.cap.get(cv2.CAP_PROP_FPS)
        return f if f and 1 < f < 1000 else None


class UsbSource(_CvLiveSource):
    """USB webcam (V4L2 on Linux / Raspberry Pi). Details in src/usbcam.py."""

    def _target(self):
        s = self.cfg["source"]
        return s["usb_device"] or s["usb_index"]          # a device path wins over the index

    def _open(self):
        self.cap = usbcam.open_capture(self._target())
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open USB camera {self._target()!r}. On a Raspberry Pi find it with "
                               f"'ls -l /dev/v4l/by-id/' and set source.usb_device")
        usbcam.configure(self.cap, self.cfg, self.target)
        usbcam.flush(self.cap)


class RtspSource(_CvLiveSource):
    def _target(self):
        return self.cfg["source"]["rtsp_url"]


class PiCameraSource(LiveSource):
    """Raspberry Pi camera through Picamera2 (cv2.VideoCapture is unreliable for CSI cameras)."""

    def _controls(self):
        p = self.cfg["picamera"]
        c = {"FrameRate": self.target or 30.0}
        if p["focus_mode"] == "manual":
            c.update({"AfMode": 0, "LensPosition": float(p["lens_position"])})
        elif p["focus_mode"] == "continuous":
            c["AfMode"] = 2
        if p["exposure_us"]:
            c.update({"AeEnable": False, "ExposureTime": int(p["exposure_us"]),
                      "AnalogueGain": float(p["gain"] or 2.0)})
        if p["awb"] == "fixed":
            c.update({"AwbEnable": False, "ColourGains": tuple(float(g) for g in p["colour_gains"])})
        return c

    def _open(self):
        from picamera2 import Picamera2    # imported here so laptops without it still work
        v = self.cfg["video"]
        controls = self._controls()
        self.cam = Picamera2()
        conf = self.cam.create_video_configuration(
            main={"size": (v["width"], v["height"]), "format": "RGB888"},   # RGB888 is BGR order = OpenCV order
            controls=controls,
        )
        self.cam.configure(conf)
        self.cam.start()
        log.info("Pi camera started with controls: %s", controls)
        time.sleep(1.0)                    # let exposure and focus settle
        try:
            md = self.cam.capture_metadata()
            log.info("Pi camera actual: exposure=%s us, gain=%s, lens=%s, colour gains=%s",
                     md.get("ExposureTime"), md.get("AnalogueGain"), md.get("LensPosition"), md.get("ColourGains"))
        except Exception:
            pass

    def _grab(self):
        return True, self.cam.capture_array("main")

    def _close(self):
        # Picamera2.close() can hang for a long time on shutdown, so run it with a time limit.
        def work():
            try:
                self.cam.stop()
            except Exception:
                pass
            try:
                self.cam.close()
            except Exception:
                pass

        th = threading.Thread(target=work, daemon=True)
        th.start()
        th.join(5)
        if th.is_alive():
            log.warning("Pi camera did not close within 5 s, continuing")


def make_source(cfg):
    t = cfg["source"]["type"]
    if t == "file":
        return FileSource(cfg)
    if t == "usb":
        return UsbSource(cfg)
    if t == "picamera":
        return PiCameraSource(cfg)
    if t == "rtsp":
        return RtspSource(cfg)
    raise ValueError(f"Unknown source type: {t}")
