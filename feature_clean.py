#!/usr/bin/env python3
"""SAE-feature the CLEAN conditioned diaries via lib_motif directly (no sae_rig dep).
diary_<arm>_clean.jsonl {turn, entry, q} -> diary_<arm>_clean-sentfeats.jsonl (per-sentence SAE).
Mirrors diary_feature.py but uses lib_motif.load_lm / load_sae_only / per_sentence_features,
because the pod has no `sae_rig` module."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="early,late,none")
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--suffix", default="_clean")
    a = ap.parse_args()

    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)
    for arm in a.arms.split(","):
        p = Path(f"diary_{arm}{a.suffix}.jsonl")
        if not p.exists():
            print(f"[feat/{arm}] MISSING {p} — skipped", flush=True)
            continue
        rows = [json.loads(l) for l in open(p)]
        entries = [dict(entry=r["turn"], text=r["entry"]) for r in rows]
        recs = L.per_sentence_features(model, tok, sae, device, entries, layer=a.layer)
        out = f"diary_{arm}{a.suffix}-sentfeats.jsonl"
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(dict(entry=r["entry"], sent=r["sent"], span=list(r["span"]),
                                        text=r["text"],
                                        feats={str(k): v for k, v in r["feats"].items()})) + "\n")
        print(f"[feat/{arm}] {len(recs)} sentences -> {out}", flush=True)


if __name__ == "__main__":
    main()
