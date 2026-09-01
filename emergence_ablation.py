#!/usr/bin/env python3
"""Emergence-ablation — the falsifiability test (Iman, 2026-09-01). GPU.

Take the Lover diary up to just BEFORE a mode is born (the scar-complex is born
at e21 -> prefix = e1..20). Generate the continuation TWICE from the identical
prefix and seed:
  * CONTROL  — normally.
  * ABLATED  — with the mode's whole FEATURE SUBSPACE projected out of the residual
    at every generation step, so the mode CANNOT light up in ANY guise (not the
    word 'scar', not the wordless reed/fire/wall register either).
We are NOT censoring a word; we are making the model representationally incapable
of the mode. If the ablated journey stays basically the SAME as control, the mode
was epiphenomenal decoration. If it DIVERGES, the mode causally reorganised the
future -> genuinely creative (rung 3). We verify the ablation actually suppressed
the complex before reading the divergence.
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
    ap.add_argument("--complex", type=int, default=3)
    ap.add_argument("--motifs", default="motifs.json")
    ap.add_argument("--prefix-entries", type=int, default=20)   # e1..20; scar births e21
    ap.add_argument("--max-new", type=int, default=3200)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--out", default="emergence_ablation.json")
    a = ap.parse_args()

    import torch
    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)

    feats = [int(f) for f in json.load(open(a.motifs))["complexes"][a.complex]["features"]]
    dirs = sae.Wd[np.asarray(feats)]                    # (k, d_model) member decoder dirs
    fset = set(feats)

    rows = L.load_diary(a.diary)
    prefix = "\n\n".join(r["text"] for r in rows if r["entry"] <= a.prefix_entries)
    ids = tok(prefix, return_tensors="pt", truncation=True, max_length=40000).input_ids.to(device)
    print(f"prefix e1..{a.prefix_entries}: {ids.shape[1]} tokens; ablating {len(feats)}-feature subspace", flush=True)

    def gen(intervene=None):
        torch.manual_seed(a.seed)
        ctx = intervene if intervene is not None else _null()
        with ctx, torch.no_grad():
            g = model.generate(ids, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)

    def complex_fire(text):
        """fraction of tokens at which ANY of the complex's features is active."""
        enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[a.layer + 1][0]
        sp = sae.encode_sparse(h)
        tot = len(sp) or 1
        hits = sum(1 for idx, val in sp
                   if fset & {int(i) for i, v in zip(idx.tolist(), val.tolist()) if v > 0})
        return hits, tot

    control = gen(None)
    ablated = gen(L.Intervention(model, a.layer, dirs, "ablate"))
    ch, ct = complex_fire(control)
    ah, at = complex_fire(ablated)

    json.dump(dict(complex=a.complex, features=feats, prefix_entries=a.prefix_entries,
                   seed=a.seed, max_new=a.max_new,
                   scar_fire_control=[ch, ct], scar_fire_ablated=[ah, at],
                   control=control, ablated=ablated), open(a.out, "w"), indent=1)
    print(f"complex {a.complex} fires: CONTROL {ch}/{ct} = {100*ch//ct}% of tokens | "
          f"ABLATED {ah}/{at} = {100*ah//max(at,1)}% (want ~0 = suppression confirmed)", flush=True)
    print("CONTROL[:400]:", control[:400].replace("\n", " "), flush=True)
    print("ABLATED[:400]:", ablated[:400].replace("\n", " "), flush=True)


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


if __name__ == "__main__":
    main()
