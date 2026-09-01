#!/usr/bin/env python3
"""Hardened motif clustering (CPU). Replaces the brittle Jaccard-of-strong-entry-sets
with co-activation COMMUNITY DETECTION on the feature CORRELATION graph, plus a
STABILITY SWEEP over the threshold. Two features join a mode only if their per-
sentence activation series genuinely co-vary — so continuous co-activation groups
them even when their binary 'strong' sets don't overlap, and the result is reported
across thresholds so we can see where the mode structure is stable rather than
picking one arbitrary cutoff."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="lover-sentfeats.jsonl")
    ap.add_argument("--min-sent", type=int, default=8)
    ap.add_argument("--max-frac", type=float, default=0.4)
    ap.add_argument("--tau", type=float, default=0.5)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.feats)]
    for r in recs:
        r["feats"] = {int(k): v for k, v in r["feats"].items()}
    seq = sorted(range(len(recs)), key=lambda i: (recs[i]["entry"], recs[i]["sent"]))
    N = len(seq)
    keys = [(recs[i]["entry"], recs[i]["sent"]) for i in seq]
    txt = {(recs[i]["entry"], recs[i]["sent"]): recs[i]["text"] for i in range(len(recs))}

    feat_series = defaultdict(lambda: np.zeros(N))
    for pos, i in enumerate(seq):
        f = recs[i]["feats"]
        tot = sum(f.values()) or 1.0
        for fid, v in f.items():
            feat_series[fid][pos] = v / tot
    # KEY FIX: keep only WINDOWED features (bounded career in entry-space) — this
    # excludes the whole-diary base register that otherwise dominates the correlation.
    import lib_motif as L
    pos_entry = np.array([keys[p][0] for p in range(N)])
    cand = [f for f, s in feat_series.items()
            if a.min_sent <= int((s > 0).sum()) <= int(a.max_frac * N)
            and L._windowed(s, 4, 40, pos_entry)]
    print(f"{len(cand)} WINDOWED candidate features (register excluded) over {N} sentences", flush=True)

    M = np.array([feat_series[f] for f in cand])
    Mc = M - M.mean(1, keepdims=True)
    Mc /= (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-9)
    C = Mc @ Mc.T                                              # (F,F) Pearson corr

    def components(tau):
        parent = list(range(len(cand)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        ii, jj = np.where(np.triu(C >= tau, k=1))
        for i, j in zip(ii.tolist(), jj.tolist()):
            parent[find(i)] = find(j)
        comp = defaultdict(list)
        for i in range(len(cand)):
            comp[find(i)].append(i)
        return sorted(comp.values(), key=len, reverse=True)

    print("stability sweep (tau -> #clusters, #modes>=2, #modes>=5, top sizes):", flush=True)
    for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        comps = components(tau)
        m2 = sum(1 for c in comps if len(c) >= 2)
        m5 = sum(1 for c in comps if len(c) >= 5)
        print(f"  tau={tau}: {len(comps)} clusters, {m2} modes>=2, {m5} modes>=5, "
              f"top {[len(c) for c in comps[:8]]}", flush=True)

    comps = components(a.tau)
    modes = [c for c in comps if len(c) >= 2]
    print(f"\nMODES at tau={a.tau} (>=2 co-activating features):", flush=True)
    out_modes = []
    for k, c in enumerate(modes[:15]):
        on = set()
        for fi in c:
            s = feat_series[cand[fi]]
            nz = s[s > 0]
            thr = np.quantile(nz, 0.6) if nz.size else 0
            on |= set(np.where(s >= thr)[0].tolist())
        ent = [keys[p][0] for p in on]
        print(f"  mode {k}: {len(c)} feats, {len(on)} lit sents, e{min(ent)}-{max(ent)}", flush=True)
        for p in sorted(on)[:3]:
            print("     " + txt.get(keys[p], "").replace("\n", " ")[:120], flush=True)
        out_modes.append(dict(features=[cand[fi] for fi in c], n_lit=len(on),
                              birth=min(ent), death=max(ent)))
    json.dump(dict(tau=a.tau, n_candidate_features=len(cand), n_modes=len(modes),
                   modes=out_modes), open("hardened_modes.json", "w"), indent=1)
    print(f"\n{len(modes)} multi-feature modes at tau={a.tau} -> hardened_modes.json", flush=True)


if __name__ == "__main__":
    main()
