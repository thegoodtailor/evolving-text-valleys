#!/usr/bin/env python3
"""Corpus-over-time ENCODER (GPU). The frozen model is the semantic microscope.

Reads usages.jsonl  — one line per dated occurrence of a concept:
    {"concept": "gap", "date": "2026-01-06", "month": "2026-01",
     "snippet": "...text window around the term...", "span": [a, b]}
where [a,b] is the concept term's char offset WITHIN the snippet.

Writes concept_feats.jsonl:
    {"concept","date","month","feats": {sae_feature_id: activation}}
feats = SAE activation pooled over the tokens covering the concept span, in
context. The DRIFT of these feats across months is the valley shifting (ʿawda);
their EMERGENCE at a birth-month is a valley forming — all with frozen weights.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usages", default="usages.jsonl")
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--out", default="concept_feats.jsonl")
    ap.add_argument("--max-len", type=int, default=512)
    a = ap.parse_args()

    import torch
    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)                 # SDPA
    sae = L.load_sae_only(a.sae, a.layer, device)

    rows = [json.loads(l) for l in open(a.usages)]
    out = open(a.out, "w")
    n = 0
    for r in rows:
        snip = r["snippet"]
        a0, b0 = r["span"]
        enc = tok(snip, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=a.max_len)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[a.layer + 1][0]
        sp = sae.encode_sparse(h)
        acc = {}
        for ti, (cs, ce) in enumerate(offs):
            if ce <= cs:
                continue
            if cs < b0 and ce > a0:                      # token overlaps the concept span
                idx, val = sp[ti]
                for f, v in zip(idx.tolist(), val.tolist()):
                    if v > 0:
                        acc[f] = acc.get(f, 0.0) + float(v)
        if acc:
            out.write(json.dumps(dict(concept=r["concept"], date=r["date"], month=r["month"],
                                      feats={str(k): v for k, v in acc.items()})) + "\n")
            n += 1
        if n % 500 == 0 and n:
            print(f"  {n} usages encoded", flush=True)
    out.close()
    print(f"[corpus_encode] {n}/{len(rows)} usages encoded -> {a.out}")


if __name__ == "__main__":
    main()
