"""Configuration: defaults, YAML loading, command-line overrides, validation."""
import copy
import logging
import os

import yaml

DEFAULTS = {
    "source": {
        "type": "file",
        "path": "samples/test_mouse.mp4",
        "usb_index": 0,
        "usb_device": None,
        "usb_fourcc": "MJPG",
        "rtsp_url": "",
        "loop": False,
        "reconnect_attempts": 3,
        "reconnect_wait_sec": 2,
    },
    "video": {
        "fps": "auto",
        "width": 640,
        "height": 480,
        "process_width": 320,
        "rotate": 0,
        "roi": None,
    },
    "detection": {
        "method": "mog2",
        "history": 500,
        "var_threshold": 25,
        "learning_rate": -1,
        "diff_threshold": 25,
        "background_image": None,
        "blur": 5,
        "morph_kernel": 5,
        "min_area": 150,
        "warmup_sec": 2,
    },
    "movement": {
        "speed_threshold": 8,
        "speed_window_sec": 0.5,
        "min_move_sec": 1.0,
        "min_still_sec": 0.5,
        "jitter_px": 1.5,
        "px_per_cm": None,
    },
    "recording": {
        "enabled": True,
        "continuous": False,
        "pre_buffer_sec": 3,
        "post_buffer_sec": 3,
        "codec": "mp4v",
        "clip_dir": "clips",
        "min_free_gb": 2,
    },
    "session": {
        "id": "auto",
        "cage_id": "cage01",
        "treatment_code": "",
        "max_duration_sec": None,
    },
    "output": {
        "dir": "output",
        "save_frame_data": True,
        "overlay_video": False,
        "show_window": False,
        "auto_graphs": True,
    },
    "picamera": {
        "focus_mode": "manual",
        "lens_position": 3.3,
        "exposure_us": None,
        "gain": None,
        "awb": "auto",
        "colour_gains": [2.0, 1.6],
    },
    "usbcam": {                  # USB webcam image controls (Linux / V4L2). null = leave the camera as it is
        "auto_exposure": None,
        "exposure": None,
        "gain": None,
        "autofocus": None,
        "focus": None,
        "auto_wb": None,
        "wb_temperature": None,
    },
    "logging": {
        "level": "INFO",
        "file": "output/run.log",
        "health_interval_sec": 60,
    },
}

SOURCE_TYPES = ("file", "usb", "picamera", "rtsp")
METHODS = ("mog2", "fixed_background", "frame_diff")
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


class ConfigError(ValueError):
    pass


def _check(cond, msg):
    if not cond:
        raise ConfigError(msg)


def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def deep_merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _check_unknown(cfg, ref, path=""):
    for k, v in cfg.items():
        _check(k in ref, f"Unknown setting: {path}{k}")
        if isinstance(v, dict) and isinstance(ref[k], dict):
            _check_unknown(v, ref[k], f"{path}{k}.")


def apply_overrides(cfg, overrides):
    """overrides: list of 'section.key=value' strings (value parsed as YAML)."""
    for item in overrides or []:
        _check("=" in item, f"Bad override '{item}', use section.key=value")
        dotted, raw = item.split("=", 1)
        parts = dotted.strip().split(".")
        _check(len(parts) == 2, f"Bad override key '{dotted}', use section.key")
        sec, key = parts
        _check(sec in DEFAULTS and key in DEFAULTS[sec], f"Unknown setting: {dotted}")
        cfg[sec][key] = yaml.safe_load(raw)
    return cfg


def load_config(path=None, overrides=None):
    user = {}
    if path:
        _check(os.path.isfile(path), f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            user = yaml.safe_load(f) or {}
        _check(isinstance(user, dict), "Config file must contain sections (source, video, ...)")
    _check_unknown(user, DEFAULTS)
    cfg = deep_merge(DEFAULTS, user)
    apply_overrides(cfg, overrides)
    validate(cfg, check_source_file=False)
    return cfg


def validate(cfg, check_source_file=True):
    s, v, d = cfg["source"], cfg["video"], cfg["detection"]
    m, r, se = cfg["movement"], cfg["recording"], cfg["session"]

    _check(s["type"] in SOURCE_TYPES, f"source.type must be one of {SOURCE_TYPES}")
    if s["type"] == "file" and check_source_file:
        _check(s["path"] and os.path.isfile(str(s["path"])), f"source.path not found: {s['path']}")
    if s["type"] == "rtsp":
        _check(bool(s["rtsp_url"]), "source.rtsp_url is required when type is rtsp")
    _check(isinstance(s["usb_index"], int), "source.usb_index must be an integer")
    _check(s["usb_device"] is None or (isinstance(s["usb_device"], str) and s["usb_device"].strip() != ""),
           "source.usb_device must be null or a device path such as /dev/video0")
    _check(s["usb_fourcc"] is None or (isinstance(s["usb_fourcc"], str) and len(s["usb_fourcc"]) == 4),
           "source.usb_fourcc must be null or 4 characters (MJPG or YUYV)")
    _check(isinstance(s["reconnect_attempts"], int) and s["reconnect_attempts"] >= 0,
           "source.reconnect_attempts must be >= 0")
    _check(_isnum(s["reconnect_wait_sec"]) and s["reconnect_wait_sec"] >= 0,
           "source.reconnect_wait_sec must be >= 0")

    _check(v["fps"] == "auto" or (_isnum(v["fps"]) and v["fps"] > 0),
           "video.fps must be 'auto' or a positive number")
    for k in ("width", "height"):
        _check(isinstance(v[k], int) and v[k] > 0, f"video.{k} must be a positive integer")
    _check(v["process_width"] is None or (isinstance(v["process_width"], int) and v["process_width"] >= 32),
           "video.process_width must be null or an integer >= 32")
    _check(v["rotate"] in (0, 90, 180, 270), "video.rotate must be 0, 90, 180 or 270")
    roi = v["roi"]
    _check(roi is None or (isinstance(roi, (list, tuple)) and len(roi) == 4
                           and all(isinstance(i, int) and i >= 0 for i in roi)
                           and roi[2] > 0 and roi[3] > 0),
           "video.roi must be null or [x, y, w, h] with whole numbers (w, h > 0)")

    _check(d["method"] in METHODS, f"detection.method must be one of {METHODS}")
    if d["method"] == "fixed_background":
        _check(d["background_image"] and os.path.isfile(str(d["background_image"])),
               "detection.background_image must be an existing image when method is fixed_background")
    _check(isinstance(d["history"], int) and d["history"] > 0, "detection.history must be a positive integer")
    _check(_isnum(d["var_threshold"]) and d["var_threshold"] > 0, "detection.var_threshold must be > 0")
    _check(_isnum(d["learning_rate"]) and (d["learning_rate"] == -1 or 0 <= d["learning_rate"] <= 1),
           "detection.learning_rate must be -1 or between 0 and 1")
    _check(_isnum(d["diff_threshold"]) and 0 < d["diff_threshold"] < 255,
           "detection.diff_threshold must be between 0 and 255")
    _check(isinstance(d["blur"], int) and (d["blur"] == 0 or d["blur"] % 2 == 1),
           "detection.blur must be 0 or an odd integer")
    _check(isinstance(d["morph_kernel"], int) and d["morph_kernel"] >= 1,
           "detection.morph_kernel must be an integer >= 1")
    _check(_isnum(d["min_area"]) and d["min_area"] >= 0, "detection.min_area must be >= 0")
    _check(_isnum(d["warmup_sec"]) and d["warmup_sec"] >= 0, "detection.warmup_sec must be >= 0")

    _check(_isnum(m["speed_threshold"]) and m["speed_threshold"] >= 0, "movement.speed_threshold must be >= 0")
    _check(_isnum(m["speed_window_sec"]) and m["speed_window_sec"] > 0, "movement.speed_window_sec must be > 0")
    _check(_isnum(m["min_move_sec"]) and m["min_move_sec"] >= 0, "movement.min_move_sec must be >= 0")
    _check(_isnum(m["min_still_sec"]) and m["min_still_sec"] >= 0, "movement.min_still_sec must be >= 0")
    _check(_isnum(m["jitter_px"]) and m["jitter_px"] >= 0, "movement.jitter_px must be >= 0")
    _check(m["px_per_cm"] is None or (_isnum(m["px_per_cm"]) and m["px_per_cm"] > 0),
           "movement.px_per_cm must be null or > 0")

    _check(isinstance(r["enabled"], bool) and isinstance(r["continuous"], bool),
           "recording.enabled and recording.continuous must be true or false")
    _check(_isnum(r["pre_buffer_sec"]) and r["pre_buffer_sec"] >= 0, "recording.pre_buffer_sec must be >= 0")
    _check(_isnum(r["post_buffer_sec"]) and r["post_buffer_sec"] >= 0, "recording.post_buffer_sec must be >= 0")
    _check(isinstance(r["codec"], str) and len(r["codec"]) == 4, "recording.codec must be 4 characters (e.g. mp4v)")
    _check(_isnum(r["min_free_gb"]) and r["min_free_gb"] >= 0, "recording.min_free_gb must be >= 0")

    _check(se["max_duration_sec"] is None or (_isnum(se["max_duration_sec"]) and se["max_duration_sec"] > 0),
           "session.max_duration_sec must be null or > 0")
    if s["type"] == "file" and s["loop"] and se["max_duration_sec"] is None:
        logging.getLogger(__name__).warning(
            "source.loop is true but session.max_duration_sec is null: the run will never stop on its own")

    o = cfg["output"]
    _check(isinstance(o["auto_graphs"], bool), "output.auto_graphs must be true or false")

    pc = cfg["picamera"]
    _check(pc["focus_mode"] in ("none", "manual", "continuous"),
           "picamera.focus_mode must be none, manual or continuous")
    _check(_isnum(pc["lens_position"]) and 0 <= pc["lens_position"] <= 15,
           "picamera.lens_position must be between 0 and 15 (dioptres; 3.3 is about 30 cm)")
    _check(pc["exposure_us"] is None or (isinstance(pc["exposure_us"], int) and pc["exposure_us"] > 0),
           "picamera.exposure_us must be null (auto) or a positive integer in microseconds")
    _check(pc["gain"] is None or (_isnum(pc["gain"]) and pc["gain"] > 0), "picamera.gain must be null or > 0")
    _check(pc["awb"] in ("auto", "fixed"), "picamera.awb must be auto or fixed")
    _check(isinstance(pc["colour_gains"], (list, tuple)) and len(pc["colour_gains"]) == 2
           and all(_isnum(g) and g > 0 for g in pc["colour_gains"]),
           "picamera.colour_gains must be [red, blue] with positive numbers")
    uc = cfg["usbcam"]
    for k in ("auto_exposure", "autofocus", "auto_wb"):
        _check(uc[k] is None or isinstance(uc[k], bool), f"usbcam.{k} must be null, true or false")
    for k in ("exposure", "gain", "focus", "wb_temperature"):
        _check(uc[k] is None or (_isnum(uc[k]) and uc[k] >= 0), f"usbcam.{k} must be null or a number >= 0")
    _check(_isnum(cfg["logging"]["health_interval_sec"]) and cfg["logging"]["health_interval_sec"] >= 0,
           "logging.health_interval_sec must be >= 0 (0 = off)")

    _check(str(cfg["logging"]["level"]).upper() in LOG_LEVELS, f"logging.level must be one of {LOG_LEVELS}")
    return cfg
