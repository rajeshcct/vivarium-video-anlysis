"""Make the movement graphs for a finished session folder.

This runs automatically at the end of a session when output.auto_graphs is true.
Use this tool to redo the graphs later, for example after setting movement.px_per_cm.

    python tools/make_graphs.py output/<session_folder>

Graphs need matplotlib (laptop: pip install matplotlib   Pi: sudo apt install python3-matplotlib).
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.graphs import render_graphs


def main():
    ap = argparse.ArgumentParser(description="Make movement graphs for a finished session")
    ap.add_argument("session_dir", help="a folder like output/<session_id>")
    ap.add_argument("--px-per-cm", type=float, default=None,
                    help="override movement.px_per_cm to draw the graphs in cm")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sd = Path(a.session_dir)
    if not (sd / "config_used.yaml").is_file():
        sys.exit(f"config_used.yaml not found in {sd}. Is this a session folder?")
    with open(sd / "config_used.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if a.px_per_cm:
        cfg["movement"]["px_per_cm"] = a.px_per_cm
    if not render_graphs(sd, cfg):
        sys.exit(1)


if __name__ == "__main__":
    main()
