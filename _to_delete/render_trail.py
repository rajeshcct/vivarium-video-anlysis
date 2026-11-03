"""Draw the mouse's movement path (a trail line) on saved clips, and make line graphs.

This runs automatically at the end of a session when output.auto_trail is true.
Use this tool to redo it later, or with different settings, on a finished session folder.
The original clips are NOT changed: trail videos go to <session>/trails/.

    python tools/render_trail.py output/<session_folder>
    python tools/render_trail.py output/<session_folder> --trail-sec 0       # trail from the start of each clip
    python tools/render_trail.py output/<session_folder> --trail-sec 10 --thickness 3
    python tools/render_trail.py output/<session_folder> --no-graphs         # clips only
    python tools/render_trail.py output/<session_folder> --no-clips          # graphs only

Graphs need matplotlib (laptop: pip install matplotlib   Pi: sudo apt install python3-matplotlib).
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from src.trail import render_session


def main():
    ap = argparse.ArgumentParser(description="Draw movement trails on clips and make graphs")
    ap.add_argument("session_dir", help="a folder like output/<session_id>")
    ap.add_argument("--trail-sec", type=float, default=5.0,
                    help="length of the visible trail in seconds (0 = from the start of the clip)")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--no-clips", action="store_true")
    ap.add_argument("--no-graphs", action="store_true")
    a = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sd = Path(a.session_dir)
    if not (sd / "config_used.yaml").is_file():
        sys.exit(f"config_used.yaml not found in {sd}. Is this a session folder?")
    with open(sd / "config_used.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not render_session(sd, cfg, a.trail_sec, a.thickness, not a.no_clips, not a.no_graphs):
        sys.exit(1)


if __name__ == "__main__":
    main()
