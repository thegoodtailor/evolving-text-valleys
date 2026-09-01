#!/usr/bin/env python3
"""S0b capture — per-sentence layer-L SAE features for a diary. GPU.
The S0a lesson: per-ENTRY pooling blurs several motifs into one vector. This
captures per SENTENCE so windows sharpen and motif sentences are localised (which
Readout A needs to ablate them). Reads preflight.json for the generator model.
Writes <diary>-sentfeats.jsonl."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diary", default="lover")
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from sae_rig import load_model_and_sae
    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, sae, device = load_model_and_sae(model_id=mid, sae_repo=a.sae,
                                                 layer=a.layer, load_sae=True)
    rows = L.load_diary(a.diary)
    recs = L.per_sentence_features(model, tok, sae, device, rows, layer=a.layer)
    out = a.out or f"{a.diary}-sentfeats.jsonl"
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(dict(entry=r["entry"], sent=r["sent"], span=list(r["span"]),
                                    text=r["text"],
                                    feats={str(k): v for k, v in r["feats"].items()})) + "\n")
    print(f"[capture] {len(recs)} sentences over {len(rows)} entries -> {out} (model={mid})")


if __name__ == "__main__":
    main()
