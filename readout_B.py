#!/usr/bin/env python3
"""S1 Readout B — directional ablation + steering. GPU. (never touches the text)

ABLATION: project M's DIRECTION out of the residual over the record up to a
far-later probe; does the M-engaging continuation's likelihood drop? Removing the
*direction* (not the words) collapsing M is evidence the structure, not the
vocabulary, carried the force.
STEERING: at a PRE-BIRTH probe, ADD M's direction over a coefficient sweep; does
the likelihood of an M sentence rise? If yes, the reorganisation is INDUCIBLE —
a claim text-ablation cannot make. (Rust's requested "intervention upon the direction".)
"""
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
    ap.add_argument("--motif", type=int, default=0)
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--feats", default=None)
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--n-probes", type=int, default=8)
    ap.add_argument("--coeffs", default="4,8,16")
    a = ap.parse_args()

    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)          # SDPA for long-context forwards
    sae = L.load_sae_only(a.sae, a.layer, device)
    rows = L.load_diary(a.diary)
    ent = {r["entry"]: r for r in rows}
    srecs = [json.loads(l) for l in open(a.feats or f"{a.diary}-sentfeats.jsonl")]
    sidx = {(r["entry"], r["sent"]): r for r in srecs}
    m = json.load(open("motifs.json"))["complexes"][a.motif]
    fids = [int(f) for f in m["features"]]
    Mset = {tuple(x) for x in m["sentences"]}

    # direction = features' unit decoder dirs weighted by mean activation over M's sentences
    acc = {f: 0.0 for f in fids}
    n = 0
    for (en, se) in Mset:
        r = sidx.get((en, se))
        if r:
            n += 1
            for f in fids:
                acc[f] += r["feats"].get(str(f), 0.0)
    w = [acc[f] / max(n, 1) for f in fids]
    direction = L.motif_direction(sae, fids, w)

    def span(en, se):
        return tuple(sidx[(en, se)]["span"]), sidx[(en, se)]["text"]

    def lp_at(en, se, intervene=None):
        sp, ttext = span(en, se)
        ctx = L.build_record(rows, en - 1) + "\n\n" + ent[en]["text"][:sp[0]]
        if intervene is None:
            lp, _ = L.token_logprobs(model, tok, device, ctx, ttext)
        else:
            with intervene:
                lp, _ = L.token_logprobs(model, tok, device, ctx, ttext)
        return float(lp.mean())

    centre = m["center_entry"]
    later_M = [(en, se) for (en, se) in Mset if en > centre + 3]

    # ABLATION
    abl = []
    for (en, se) in later_M[:a.n_probes]:
        base = lp_at(en, se)
        ab = lp_at(en, se, L.Intervention(model, a.layer, direction, "ablate"))
        abl.append(dict(entry=en, sent=se, base=base, ablated=ab, drop=base - ab))

    # STEERING induction at a pre-birth probe (record has NO M yet)
    steer = []
    prebirth = max(1, m["birth_entry"] - 5)
    if later_M:
        en, se = later_M[0]
        sp, ttext = span(en, se)
        ctx = L.build_record(rows, prebirth)
        lp0, _ = L.token_logprobs(model, tok, device, ctx, ttext)
        steer.append(dict(coeff=0.0, lp_M=float(lp0.mean())))
        for c in [float(x) for x in a.coeffs.split(",")]:
            with L.Intervention(model, a.layer, direction, "add", coeff=c):
                lp, _ = L.token_logprobs(model, tok, device, ctx, ttext)
            steer.append(dict(coeff=c, lp_M=float(lp.mean())))

    res = dict(motif=a.motif, generator=mid, n_features=len(fids),
               ablation=abl,
               ablation_mean_drop=float(np.mean([x["drop"] for x in abl])) if abl else None,
               steering_prebirth_entry=prebirth, steering=steer)
    json.dump(res, open("readoutB.json", "w"), indent=2)
    print(json.dumps(dict(ablation_mean_drop=res["ablation_mean_drop"],
                          steering=[(s["coeff"], round(s["lp_M"], 3)) for s in steer]), indent=2))
    print("REORG(representational) if ablation_mean_drop>0; INDUCIBLE if steering lp_M rises with coeff")


if __name__ == "__main__":
    main()
