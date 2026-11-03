"""Compare the tracker's timeline.csv with a truth file (from make_test_video.py).

    python tools/compare_truth.py --truth samples/synthetic.truth.csv --timeline output/<session>/timeline.csv

Only the time ranges listed in the truth file are compared. Agreement is measured every 0.1 s.
"""
import argparse
import csv


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [(float(r["start_s"]), float(r["end_s"]), r["state"]) for r in csv.DictReader(f)]


def state_at(runs, t):
    for s, e, st in runs:
        if s <= t < e:
            return st
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--step", type=float, default=0.1)
    a = ap.parse_args()

    truth, measured = read(a.truth), read(a.timeline)
    total = same = 0
    t_mov = t_still = m_mov = m_still = 0.0
    wrong = []
    for s, e, st in truth:
        t = s
        while t < e:
            m = state_at(measured, t)
            if m is not None:
                total += 1
                if m == st:
                    same += 1
                else:
                    wrong.append((round(t, 1), st, m))
                if st == "moving":
                    t_mov += a.step
                    m_mov += a.step if m == "moving" else 0
                else:
                    t_still += a.step
                    m_still += a.step if m == "still" else 0
            t += a.step

    if not total:
        print("No overlapping time between truth and timeline.")
        return
    print(f"Agreement          : {100 * same / total:.1f}%  ({same}/{total} samples)")
    print(f"Truly moving time  : {t_mov:.1f}s, tracker agreed on {m_mov:.1f}s")
    print(f"Truly still time   : {t_still:.1f}s, tracker agreed on {m_still:.1f}s")
    if wrong:
        # group consecutive wrong samples into ranges
        ranges, start, prev = [], wrong[0], wrong[0]
        for w in wrong[1:]:
            if w[0] - prev[0] > a.step * 1.5 or w[1:] != prev[1:]:
                ranges.append((start, prev))
                start = w
            prev = w
        ranges.append((start, prev))
        print("\nDisagreements (time range, truth -> tracker):")
        for s, e in ranges[:15]:
            print(f"  {s[0]:6.1f}s - {e[0]:6.1f}s   {s[1]} -> {s[2]}")
        if len(ranges) > 15:
            print(f"  ... and {len(ranges) - 15} more")


if __name__ == "__main__":
    main()
