"""Merge every output/<session>/summary.csv into one comparison table.

    python tools/summarize.py                  # reads ./output
    python tools/summarize.py --dir output --out output/all_summaries.csv
"""
import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="output")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = Path(args.dir)
    files = sorted(base.glob("*/summary.csv"))
    if not files:
        print(f"No summary.csv files found under {base}")
        return

    rows, fields = [], []
    for f in files:
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
                for k in row:
                    if k not in fields:
                        fields.append(k)

    out = Path(args.out) if args.out else base / "all_summaries.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    show = ["session_id", "cage_id", "treatment_code", "session_sec", "moving_sec",
            "still_sec", "pct_moving", "moving_bouts", "longest_still_sec", "mouse_found_pct"]
    show = [k for k in show if k in fields]
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in show}
    print("  ".join(k.ljust(widths[k]) for k in show))
    for r in rows:
        print("  ".join(str(r.get(k, "")).ljust(widths[k]) for k in show))
    print(f"\nMerged {len(rows)} sessions into {out}")


if __name__ == "__main__":
    main()
