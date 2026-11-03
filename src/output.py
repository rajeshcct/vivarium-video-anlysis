"""Per-session output files."""
import csv
from pathlib import Path

import yaml


class SessionOutput:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = Path(cfg["output"]["dir"]) / cfg["session"]["id"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta = {
            "session_id": cfg["session"]["id"],
            "cage_id": cfg["session"]["cage_id"],
            "treatment_code": cfg["session"]["treatment_code"],
        }
        with open(self.dir / "config_used.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        self._frames = None
        self._n = 0
        if cfg["output"]["save_frame_data"]:
            self._ff = open(self.dir / "frame_data.csv", "w", newline="", encoding="utf-8")
            self._frames = csv.writer(self._ff)
            self._frames.writerow(["session_id", "t_s", "x_px", "y_px", "found", "speed_px_s", "moving"])

    def add_frame(self, res):
        if self._frames is None:
            return
        self._frames.writerow([
            self.meta["session_id"], round(res["t"], 3),
            "" if res["x"] is None else round(res["x"], 1),
            "" if res["y"] is None else round(res["y"], 1),
            int(res["found"]), round(res["speed"], 2), int(res["moving"]),
        ])
        self._n += 1
        if self._n % 100 == 0:
            self._ff.flush()

    def write_timeline(self, runs):
        with open(self.dir / "timeline.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["session_id", "cage_id", "treatment_code", "start_s", "end_s", "duration_s", "state"])
            for s, e, st in runs:
                w.writerow([self.meta["session_id"], self.meta["cage_id"], self.meta["treatment_code"],
                            round(s, 2), round(e, 2), round(e - s, 2), st])

    def write_summary(self, summary):
        with open(self.dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary.keys()))
            w.writeheader()
            w.writerow(summary)

    def close(self):
        if self._frames is not None:
            self._ff.close()
