#!/usr/bin/env python3
"""S1 Readout A — formation-ablation counterfactual likelihood. GPU.

Does removing motif M's FORMATION sentences (its early instances) make FAR-LATER
M-engaging text improbable? Per-token log-prob over hundreds of tokens (the power
fix). Reorganisation-vs-priming discriminators, all shipped:
  * formation ablation with a far-later readout (not adjacent) — priming is local;
  * degradation control — M-neutral later text must NOT move;
  * matched control (Elenchos) — dropping equal-count non-M early sentences must NOT move it.
REORG signal: delta_M_formation  >>  delta_M_matched  AND  >>  delta_neutral.
"""
import argparse
import json
import random
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
    ap.add_argument("--max-targets", type=int, default=12)
    a = ap.parse_args()

    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)          # SDPA — no SAE needed for likelihood

    rows = L.load_diary(a.diary)
    ent = {r["entry"]: r for r in rows}
    srecs = [json.loads(l) for l in open(a.feats or f"{a.diary}-sentfeats.jsonl")]
    sidx = {(r["entry"], r["sent"]): r for r in srecs}
    m = json.load(open("motifs.json"))["complexes"][a.motif]
    Mset = {tuple(x) for x in m["sentences"]}
    centre = m["center_entry"]

    def span(en, se):
        return tuple(sidx[(en, se)]["span"]), sidx[(en, se)]["text"]

    formation = [(en, se) for (en, se) in Mset if en < centre]
    later_M = [(en, se) for (en, se) in Mset if en > centre + 3][:a.max_targets]
    later_ent = {en for (en, se) in later_M}
    neutral = [(r["entry"], r["sent"]) for r in srecs
               if r["entry"] in later_ent and (r["entry"], r["sent"]) not in Mset][:len(later_M)]
    early_ent = {en for (en, se) in formation}
    nonM_early = [(r["entry"], r["sent"]) for r in srecs
                  if r["entry"] in early_ent and (r["entry"], r["sent"]) not in Mset]
    random.seed(20260901)
    random.shuffle(nonM_early)
    matched = nonM_early[:len(formation)]

    form_spans = [(en,) + span(en, se)[0] for (en, se) in formation]
    matched_spans = [(en,) + span(en, se)[0] for (en, se) in matched]

    def deltas(targets, drop):
        d = []
        for (en, se) in targets:
            sp, ttext = span(en, se)
            ctx_i = L.build_record(rows, en - 1) + "\n\n" + ent[en]["text"][:sp[0]]
            ctx_a = L.build_record(rows, en - 1, drop_spans=drop) + "\n\n" + ent[en]["text"][:sp[0]]
            lp_i, _ = L.token_logprobs(model, tok, device, ctx_i, ttext)
            lp_a, _ = L.token_logprobs(model, tok, device, ctx_a, ttext)
            d.append(float(lp_i.mean() - lp_a.mean()))
        return d

    res = dict(motif=a.motif, generator=mid,
               n_formation=len(formation), n_later_M=len(later_M), n_neutral=len(neutral),
               delta_M_formation=deltas(later_M, form_spans),
               delta_neutral_formation=deltas(neutral, form_spans),
               delta_M_matched=deltas(later_M, matched_spans))
    for k in [k for k in list(res) if k.startswith("delta")]:
        res[k + "_mean"] = float(np.mean(res[k])) if res[k] else None
    json.dump(res, open("readoutA.json", "w"), indent=2)
    summ = {k: res[k] for k in res if k.endswith("_mean") or k.startswith("n_")}
    print(json.dumps(summ, indent=2))
    print("REORG(behavioural) if delta_M_formation_mean >> delta_M_matched_mean AND >> delta_neutral_formation_mean")


if __name__ == "__main__":
    main()
