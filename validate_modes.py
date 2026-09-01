#!/usr/bin/env python3
"""Triple-validate candidate modes — the STEERING leg (GPU). For each candidate
complex, build its direction from member SAE features, steer it (add) at a
pre-birth probe, and check whether the mode's own sentence becomes MORE LIKELY as
the coefficient rises. Monotonic rise = an inducible, steerable mode (validated);
flat/falling = not a real steerable mode. One model load for all candidates."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diary", default="lover")
    ap.add_argument("--complexes", default="225,406,2,3")
    ap.add_argument("--motifs", default="motifs.json")
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--coeffs", default="4,8,16")
    a = ap.parse_args()

    import torch  # noqa
    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)
    rows = L.load_diary(a.diary)
    srecs = [json.loads(l) for l in open(f"{a.diary}-sentfeats.jsonl")]
    sidx = {(r["entry"], r["sent"]): r for r in srecs}
    m = json.load(open(a.motifs))["complexes"]
    coeffs = [float(x) for x in a.coeffs.split(",")]

    results = {}
    for ci in [int(x) for x in a.complexes.split(",")]:
        c = m[ci]
        feats = [int(f) for f in c["features"]]
        direction = L.motif_direction(sae, feats, np.ones(len(feats)))
        Mset = sorted(tuple(s) for s in c["sentences"])
        en, se = Mset[len(Mset) // 2]
        ttext = sidx[(en, se)]["text"]
        prebirth = max(1, c["birth_entry"] - 5)
        ctx = L.build_record(rows, prebirth)
        base, _ = L.token_logprobs(model, tok, device, ctx, ttext)
        lps = [(0.0, float(base.mean()))]
        for cf in coeffs:
            with L.Intervention(model, a.layer, direction, "add", coeff=cf):
                lp, _ = L.token_logprobs(model, tok, device, ctx, ttext)
            lps.append((cf, float(lp.mean())))
        rising = lps[-1][1] > lps[0][1]
        results[ci] = dict(n_feats=len(feats), steering=lps, inducible=rising,
                           delta=lps[-1][1] - lps[0][1])
        print(f"complex {ci} ({len(feats)} feats): steering lp "
              f"{[round(x[1], 3) for x in lps]} | inducible={rising} "
              f"(Δ={lps[-1][1] - lps[0][1]:+.3f})", flush=True)

    json.dump(results, open("validate_modes.json", "w"), indent=1)
    print("VALIDATED (inducible) modes:", [ci for ci, r in results.items() if r["inducible"]], flush=True)


if __name__ == "__main__":
    main()
