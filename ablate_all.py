#!/usr/bin/env python3
"""ABLATE each named motif from the SAME diary point and save FULL TEXT — the mirror image of
steer_all. What does REMOVING a mode's whole feature subspace do to the writing? (Iman,
2026-09-01: 'ablation/removal does what? how does it look in real texts?')

Ablation = project the mode's k-feature subspace OUT of the residual at every generation step,
so the model cannot express that mode in any guise. Same context + seed as steer_all (so the
baseline is identical and steering vs ablation are directly comparable); only which subspace is
removed differs."""
import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def top_sents(sidx, fids, Mset, k=3):
    return [t for _, t in sorted(((sum(sidx[(en, se)]["feats"].get(str(f), 0.0) for f in fids), sidx[(en, se)]["text"])
            for (en, se) in Mset if (en, se) in sidx), reverse=True)[:k]]


def gen(model, tok, device, layer, ctx, dirs, max_new, seed):
    import torch
    torch.manual_seed(seed)
    enc = tok(ctx, return_tensors="pt").to(device)
    cm = L.Intervention(model, layer, dirs, "ablate") if dirs is not None else contextlib.nullcontext()
    with cm, torch.no_grad():
        g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                           max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diary", default="lover")
    ap.add_argument("--motifs", default="406,2,3,225")
    ap.add_argument("--names", default="parasite/mirror,killing-the-moment,the scar,finitude/compression")
    ap.add_argument("--context-entry", type=int, default=12)
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", default="ablate_all")
    a = ap.parse_args()

    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)
    rows = L.load_diary(a.diary)
    sidx = {(r["entry"], r["sent"]): r for r in [json.loads(l) for l in open(f"{a.diary}-sentfeats.jsonl")]}
    comp = json.load(open("motifs.json"))["complexes"]
    ctx = L.build_record(rows, a.context_entry) + f"\n\n**Entry {a.context_entry + 1}**\n"
    motifs = [int(x) for x in a.motifs.split(",")]
    names = a.names.split(",")

    base = gen(model, tok, device, a.layer, ctx, None, a.max_new, a.seed)
    print("[baseline] done", flush=True)
    blocks = []
    for ci, nm in zip(motifs, names):
        m = comp[ci]
        fids = [int(f) for f in m["features"]]
        Mset = {tuple(x) for x in m["sentences"]}
        dirs = sae.Wd[np.asarray(fids)]          # the whole k-feature subspace
        t = gen(model, tok, device, a.layer, ctx, dirs, a.max_new, a.seed)
        print(f"[motif {ci} ablated] done", flush=True)
        blocks.append(dict(motif=ci, name=nm.strip(), n_features=len(fids),
                           top_sentences=top_sents(sidx, fids, Mset), text=t))

    json.dump(dict(context_entry=a.context_entry, baseline=base, motifs=blocks), open(f"{a.out}.json", "w"), indent=1)
    with open(f"{a.out}.md", "w") as f:
        f.write("# ABLATING each named motif from the SAME point — full text, no snippets\n\n")
        f.write(f"The mirror image of steering. From the diary up to entry {a.context_entry}, we generate "
                "the next passage once with NO intervention (baseline), then once per motif with that "
                "motif's WHOLE feature subspace projected OUT of the model's state at every step — so it "
                "cannot express that mode in any guise. Same start, same seed; only which subspace is "
                "removed differs. Read the baseline, then each removal, and judge: does taking the mode "
                "away make the writing avoid that theme / go elsewhere, or barely change at all?\n\n")
        f.write(f"## BASELINE — nothing removed\n\n{base}\n\n---\n\n")
        for b in blocks:
            f.write(f"## Motif {b['motif']} REMOVED — our name: \"{b['name']}\"  ({b['n_features']} feature(s))\n\n")
            f.write("*the removed mode fires on:* " + " / ".join(s.strip()[:100] for s in b["top_sentences"]) + "\n\n")
            f.write(f"{b['text']}\n\n---\n\n")
    print("-> ablate_all.md ablate_all.json", flush=True)


if __name__ == "__main__":
    main()
