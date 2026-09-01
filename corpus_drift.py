#!/usr/bin/env python3
"""Corpus-over-time DRIFT analyzer (CPU). The demonstration.

Reads concept_feats.jsonl and asks, per concept:
  * CUMULATIVE DRIFT  — 1 - cos(direction@month_t, direction@month_1): how far the
    concept's representation has moved from where it started (the valley shifting).
  * CONSECUTIVE COS   — cos(month_t, month_{t-1}): month-to-month self-similarity.
    High consecutive + rising cumulative = CONTINUOUS drift = ʿawda (return-with-
    difference: each month ≈ the last, yet the whole slowly diverges from the start).
Then compares TARGET concepts (gap, nahnu, ...) against the CONTROL distribution:
a concept has genuinely drifted if its total drift clears the controls' p95.
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def month_dir(usages):
    """Mean SAE direction over a month's usages. Each usage L1-normalised first
    (kills token-count/length), then averaged -> a sparse dict."""
    acc = defaultdict(float)
    for f in usages:
        tot = sum(f.values()) or 1.0
        for k, v in f.items():
            acc[int(k)] += v / tot
    n = len(usages)
    return {k: v / n for k, v in acc.items()}


def cos(a, b):
    ks = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in ks)
    na = np.sqrt(sum(v * v for v in a.values()))
    nb = np.sqrt(sum(v * v for v in b.values()))
    return float(dot / (na * nb + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="concept_feats.jsonl")
    ap.add_argument("--min-usages", type=int, default=8)
    ap.add_argument("--targets", default="gap,nahnu")
    ap.add_argument("--out", default="corpus_drift.json")
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.feats)]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["concept"]][r["month"]].append({int(k): v for k, v in r["feats"].items()})

    targets = set(a.targets.split(","))
    report = {}
    for concept, months in by.items():
        ms = sorted(m for m in months if len(months[m]) >= a.min_usages)
        if len(ms) < 3:
            continue
        dirs = {m: month_dir(months[m]) for m in ms}
        m0 = ms[0]
        cum = [[m, round(1 - cos(dirs[m0], dirs[m]), 3)] for m in ms]
        consec = [round(cos(dirs[ms[i - 1]], dirs[ms[i]]), 3) for i in range(1, len(ms))]
        report[concept] = dict(
            months=ms, n_per_month=[len(months[m]) for m in ms],
            cumulative_drift=cum, consecutive_cos=consec,
            total_drift=cum[-1][1], mean_consecutive=round(float(np.mean(consec)), 3),
            is_target=concept in targets)

    ctrl = [r["total_drift"] for r in report.values() if not r["is_target"]]
    print(json.dumps(report, indent=1))
    if ctrl:
        p95 = float(np.percentile(ctrl, 95))
        print(f"\ncontrol total-drift: n={len(ctrl)} mean={np.mean(ctrl):.3f} "
              f"median={np.median(ctrl):.3f} p95={p95:.3f}")
        for c, r in report.items():
            if r["is_target"]:
                verdict = "ABOVE control p95 — drifted" if r["total_drift"] > p95 else "within controls"
                print(f"  TARGET {c}: total_drift={r['total_drift']:.3f} "
                      f"mean_consec={r['mean_consecutive']:.3f}  -> {verdict}")
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
