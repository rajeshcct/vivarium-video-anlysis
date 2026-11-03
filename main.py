"""Mouse movement tracker: entry point.

Examples:
    python main.py                                    # config.yaml, one session
    python main.py --input samples/                   # every video in a folder
    python main.py --config config_pi.yaml            # Raspberry Pi camera
    python main.py --set detection.min_area=200 --set movement.speed_threshold=12
"""
import argparse
import copy
import datetime
import logging
import shutil
import signal
import sys
import time
from pathlib import Path

import cv2

from src.config import ConfigError, load_config, validate
from src.detector import MotionDetector, draw_overlay
from src.output import SessionOutput
from src.recorder import ClipRecorder
from src.source import make_source
from src.timeline import frames_to_runs, merge_short, summarize
from src.graphs import render_graphs

log = logging.getLogger("main")
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}


def setup_logging(cfg):
    lg = cfg["logging"]
    Path(lg["file"]).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, lg["level"].upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(lg["file"], encoding="utf-8")],
    )


def resolve_session_id(cfg, stem=None):
    sid = cfg["session"]["id"]
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # a timestamp is always included so a re-run never overwrites earlier results
    if sid == "auto":
        cfg["session"]["id"] = f"{stem or cfg['session']['cage_id']}_{stamp}"
    elif stem:
        cfg["session"]["id"] = f"{sid}_{stem}"


def check_disk(cfg):
    r = cfg["recording"]
    if not r["enabled"]:
        return
    target = Path(r["clip_dir"]) if Path(r["clip_dir"]).is_absolute() else Path(cfg["output"]["dir"])
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free / 1e9
    if free_gb < r["min_free_gb"]:
        raise RuntimeError(f"Only {free_gb:.1f} GB free on {probe}; need {r['min_free_gb']} GB "
                           f"(recording.min_free_gb). Free some space or lower the limit.")


def _on_sigterm(signum, frame):
    """kill: behave like Ctrl+C so clips and result files are closed properly."""
    raise KeyboardInterrupt


def release_source(source):
    """Release the camera without letting a hang or a second Ctrl+C stop the results from being saved."""
    try:
        source.release()
    except KeyboardInterrupt:
        log.warning("Camera release interrupted, continuing to save results")
    except Exception as e:
        log.warning("Camera release failed: %s", e)


def read_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def run_session(cfg):
    validate(cfg)
    check_disk(cfg)
    out = SessionOutput(cfg)
    source = make_source(cfg)
    log.info("Session %s | source=%s | fps=%.2f | output=%s",
             cfg["session"]["id"], cfg["source"]["type"], source.fps, out.dir)

    detector = recorder = overlay = None
    rows, last_t = [], 0.0
    prev_t, dropped, n_frames = None, 0, 0
    live = cfg["source"]["type"] != "file"
    health_iv = cfg["logging"]["health_interval_sec"]
    last_health = time.monotonic()
    max_dur = cfg["session"]["max_duration_sec"]
    want_vis = cfg["output"]["overlay_video"] or cfg["output"]["show_window"]
    started = datetime.datetime.now().isoformat(timespec="seconds")

    try:
        while True:
            ok, frame, t = source.read()
            if not ok:
                break
            if max_dur is not None and t > max_dur:
                break
            if live and prev_t is not None and (t - prev_t) > 1.5 / source.fps:
                dropped += 1
            prev_t = last_t = t
            n_frames += 1

            if detector is None:
                detector = MotionDetector(cfg, frame.shape)
                if cfg["recording"]["enabled"]:
                    h, w = frame.shape[:2]
                    recorder = ClipRecorder(cfg, source.fps, (w, h), out.dir)

            res = detector.process(frame, t)
            rows.append((t, res["x"], res["y"], res["found"], res["speed"], res["moving"]))
            out.add_frame(res)
            if recorder:
                recorder.update(frame, t, res["moving"])

            if health_iv and time.monotonic() - last_health >= health_iv:
                el = time.monotonic() - last_health
                free_gb = shutil.disk_usage(out.dir).free / 1e9
                temp = read_cpu_temp()
                log.info("health: %.1f fps | frame gaps %d | cpu %s | free disk %.1f GB | clips %d",
                         n_frames / el, dropped, f"{temp:.1f}C" if temp else "n/a", free_gb,
                         recorder.n if recorder else 0)
                n_frames, dropped, last_health = 0, 0, time.monotonic()
                if recorder and free_gb < cfg["recording"]["min_free_gb"]:
                    log.error("Free disk below %.1f GB: stopping and saving results",
                              cfg["recording"]["min_free_gb"])
                    break

            if want_vis:
                vis = draw_overlay(res)
                if cfg["output"]["overlay_video"]:
                    if overlay is None:
                        vh, vw = vis.shape[:2]
                        overlay = cv2.VideoWriter(str(out.dir / "overlay.mp4"),
                                                  cv2.VideoWriter_fourcc(*cfg["recording"]["codec"]),
                                                  source.fps, (vw, vh))
                    overlay.write(vis)
                if cfg["output"]["show_window"]:
                    cv2.imshow("tracker (q to stop)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("Stopped with q")
                        break
    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C), saving results. Please wait a few seconds and do not press Ctrl+C again")
    finally:
        if recorder:
            recorder.close(last_t)
        if overlay:
            overlay.release()
        release_source(source)
        out.close()
        if cfg["output"]["show_window"]:
            cv2.destroyAllWindows()

    if not rows:
        raise RuntimeError("No frames were read from the source")

    runs = merge_short(frames_to_runs([r[0] for r in rows], [r[5] for r in rows]),
                       cfg["movement"]["min_move_sec"], cfg["movement"]["min_still_sec"])
    meta = dict(out.meta)
    meta.update({
        "source": cfg["source"]["path"] if cfg["source"]["type"] == "file" else cfg["source"]["type"],
        "started_at": started,
        "fps": source.fps,
    })
    summary = summarize(rows, runs, cfg, meta)
    out.write_timeline(runs)
    out.write_summary(summary)

    log.info("Done: total %.0fs | moving %.0fs (%.1f%%) | still %.0fs | bouts %d | mouse found in %.1f%% of frames",
             summary["session_sec"], summary["moving_sec"], summary["pct_moving"],
             summary["still_sec"], summary["moving_bouts"], summary["mouse_found_pct"])
    if summary["mouse_found_pct"] < 80:
        log.warning("Mouse was found in less than 80% of frames: check min_area, method, ROI and lighting")

    if cfg["output"]["auto_graphs"]:
        log.info("Making graphs, please wait (Ctrl+C is ignored until they are done)")
        old_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)   # an extra Ctrl+C must not cancel the graphs
        try:
            render_graphs(out.dir, cfg)
        except Exception as e:
            log.warning("Making graphs failed (your results are safe). Run later: "
                        "python tools/make_graphs.py %s | reason: %s", out.dir, e)
        finally:
            signal.signal(signal.SIGINT, old_sigint)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Mouse movement tracker")
    ap.add_argument("--config", default="config.yaml", help="path to YAML config (default config.yaml)")
    ap.add_argument("--input", help="video file or folder of videos (overrides source.path)")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value",
                    help="override a setting, can be repeated")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_sigterm)

    try:
        cfg = load_config(args.config if Path(args.config).is_file() else None, args.set)
        setup_logging(cfg)
        if not Path(args.config).is_file():
            log.warning("Config %s not found, using defaults", args.config)

        jobs = []
        if args.input:
            if cfg["source"]["type"] != "file":
                raise ConfigError("--input can only be used with source.type = file")
            p = Path(args.input)
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXT) if p.is_dir() else [p]
            if not files:
                raise ConfigError(f"No video files found in {p}")
            for f in files:
                c = copy.deepcopy(cfg)
                c["source"]["path"] = str(f)
                resolve_session_id(c, stem=f.stem)
                jobs.append(c)
        else:
            c = copy.deepcopy(cfg)
            stem = Path(c["source"]["path"]).stem if c["source"]["type"] == "file" else None
            resolve_session_id(c, stem=stem if c["session"]["id"] == "auto" else None)
            jobs.append(c)

        failed = 0
        for c in jobs:
            try:
                run_session(c)
            except (RuntimeError, ConfigError) as e:
                failed += 1
                log.error("Session %s failed: %s", c["session"]["id"], e)
        if failed:
            sys.exit(1)
    except ConfigError as e:
        print(f"Config error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
