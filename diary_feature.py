#!/usr/bin/env python3
"""SAE-feature the conditioned diaries (GPU) so valley detection runs offline.
diary_<arm>.jsonl {turn, entry, q} -> diary_<arm>-sentfeats.jsonl (per-sentence SAE)."""
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
    a = ap.parse_args()

    from sae_rig import load_model_and_sae
    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, sae, device = load_model_and_sae(model_id=mid, sae_repo=a.sae,
                                                 layer=a.layer, load_sae=True)
    for arm in a.arms.split(","):
        p = Path(f"diary_{arm}.jsonl")
        if not p.exists():
            print(f"[feature/{arm}] MISSING {p} — skipped")
            continue
        rows = [json.loads(l) for l in open(p)]
        entries = [dict(entry=r["turn"], text=r["entry"]) for r in rows]
        recs = L.per_sentence_features(model, tok, sae, device, entries, layer=a.layer)
        out = f"diary_{arm}-sentfeats.jsonl"
        with open(out, "w") as f:
            for r in recs:
                f.write(json.dumps(dict(entry=r["entry"], sent=r["sent"], span=list(r["span"]),
                                        text=r["text"],
                                        feats={str(k): v for k, v in r["feats"].items()})) + "\n")
        print(f"[feature/{arm}] {len(recs)} sentences -> {out}")


if __name__ == "__main__":
    main()
