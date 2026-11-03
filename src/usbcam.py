"""USB webcam helpers, used by UsbSource in source.py.

On Linux (including the Raspberry Pi) the camera is opened through V4L2. Most webcams only give full
frame rate at HD size in MJPG format, so MJPG is requested by default (source.usb_fourcc). Image
controls such as exposure and focus are changed only when they are set in the usbcam section of the
config; null means "leave the camera as it is".
"""
import logging
import sys

import cv2

log = logging.getLogger(__name__)


def open_capture(target):
    """target = device index (0) or device path ('/dev/video0', '/dev/v4l/by-id/...')."""
    if sys.platform.startswith("linux"):
        return cv2.VideoCapture(target, cv2.CAP_V4L2)
    return cv2.VideoCapture(target)


def _fourcc_text(cap):
    code = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return text if code and text.isprintable() else "?"


def _set(cap, name, prop, value):
    if not cap.set(prop, value):
        log.warning("USB camera did not accept %s=%s (not supported by this camera or driver)", name, value)
        return False
    return True


def configure(cap, cfg, fps_target):
    """Apply format, size, frame rate and the optional image controls, then log what the camera really gives."""
    s, v, u = cfg["source"], cfg["video"], cfg["usbcam"]
    if s["usb_fourcc"]:
        _set(cap, "format", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*s["usb_fourcc"]))   # before the size
    _set(cap, "width", cv2.CAP_PROP_FRAME_WIDTH, v["width"])
    _set(cap, "height", cv2.CAP_PROP_FRAME_HEIGHT, v["height"])
    if fps_target:
        _set(cap, "fps", cv2.CAP_PROP_FPS, fps_target)

    # optional fixed image controls (V4L2 values). Auto exposure / focus / white balance change the image by
    # themselves and look like movement, so fix them for an experiment.
    if u["auto_exposure"] is not None:
        _set(cap, "auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE, 3 if u["auto_exposure"] else 1)   # V4L2: 3 auto, 1 manual
    if u["exposure"] is not None:
        _set(cap, "exposure", cv2.CAP_PROP_EXPOSURE, u["exposure"])                              # V4L2 unit: 0.1 ms
    if u["gain"] is not None:
        _set(cap, "gain", cv2.CAP_PROP_GAIN, u["gain"])
    if u["autofocus"] is not None:
        _set(cap, "autofocus", cv2.CAP_PROP_AUTOFOCUS, 1 if u["autofocus"] else 0)
    if u["focus"] is not None:
        _set(cap, "focus", cv2.CAP_PROP_FOCUS, u["focus"])
    if u["auto_wb"] is not None:
        _set(cap, "auto_wb", cv2.CAP_PROP_AUTO_WB, 1 if u["auto_wb"] else 0)
    if u["wb_temperature"] is not None:
        _set(cap, "wb_temperature", cv2.CAP_PROP_WB_TEMPERATURE, u["wb_temperature"])

    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    log.info("USB camera: %dx%d, %.1f fps reported, format %s", w, h, fps or 0.0, _fourcc_text(cap))
    if (w, h) != (v["width"], v["height"]):
        log.warning("USB camera gives %dx%d, not the requested %dx%d. video.roi must use the real size %dx%d",
                    w, h, v["width"], v["height"], w, h)
    return {"width": w, "height": h, "fps": fps}


def flush(cap, n=5):
    """Read and drop a few frames so exposure and the new settings have taken effect."""
    for _ in range(n):
        cap.read()
