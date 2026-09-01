#!/usr/bin/env python3
"""Emergence-ablation v2 (GPU) — the CORRECTED falsifiability test (Iman, 2026-09-01).

Prefix = Lover e1..20 (the scar-complex is born e21). Per seed, generate three
continuations from the IDENTICAL prefix+seed:
  * control       — no intervention
  * scar-ablated  — the scar complex's 7-feature SUBSPACE projected out at every step
  * random-ablated— a matched RANDOM 7-feature subspace projected out (the control
                    for "any ablation perturbs generation")
Fixes over v1: suppression is verified by POOLED scar-mass (fraction of the
continuation's SAE activation landing on the complex's features), not per-token
top-k; and the random-subspace arm is the missing matched control.

VERDICT — the scar-mode causally reorganises the future (genuinely creative, rung 3)
iff BOTH: (a) scar_mass(scar-ablated) << scar_mass(control)  [suppression real], AND
(b) div_scar > div_rand  [ablating the mode moves the trajectory MORE than random].
If div_scar ≈ div_rand, the mode is decoration. Saves ALL continuations + metrics.
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diary", default="lover")
    ap.add_argument("--complex", type=int, default=3)
    ap.add_argument("--motifs", default="motifs.json")
    ap.add_argument("--prefix-entries", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=2000)
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--out", default="emergence_v2.json")
    a = ap.parse_args()

    import torch
    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)

    feats = [int(f) for f in json.load(open(a.motifs))["complexes"][a.complex]["features"]]
    scar_dirs = sae.Wd[np.asarray(feats)]
    rng = np.random.default_rng(20260901)
    rand_feats = [int(x) for x in rng.choice(sae.d_sae, size=len(feats), replace=False)]
    rand_dirs = sae.Wd[np.asarray(rand_feats)]

    rows = L.load_diary(a.diary)
    prefix = "\n\n".join(r["text"] for r in rows if r["entry"] <= a.prefix_entries)
    ids = tok(prefix, return_tensors="pt", truncation=True, max_length=40000).input_ids.to(device)
    print(f"prefix e1..{a.prefix_entries}: {ids.shape[1]} tok; scar feats {feats}; random feats {rand_feats}", flush=True)

    def gen(seed, intervene=None):
        torch.manual_seed(seed)
        ctx = intervene if intervene is not None else _null()
        with ctx, torch.no_grad():
            g = model.generate(ids, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id)
        return tok.decode(g[0][ids.shape[1]:], skip_special_tokens=True)

    def analyze(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=4096).to(device)
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[a.layer + 1][0]
        pooled = {}
        for idx, val in sae.encode_sparse(h):
            for f, v in zip(idx.tolist(), val.tolist()):
                if v > 0:
                    pooled[f] = pooled.get(f, 0.0) + v
        tot = sum(pooled.values()) or 1.0
        return pooled, sum(pooled.get(f, 0.0) for f in feats) / tot

    def cos(x, y):
        ks = set(x) | set(y)
        dot = sum(x.get(k, 0.0) * y.get(k, 0.0) for k in ks)
        nx = np.sqrt(sum(v * v for v in x.values()))
        ny = np.sqrt(sum(v * v for v in y.values()))
        return dot / (nx * ny + 1e-9)

    results = []
    for s in [int(x) for x in a.seeds.split(",")]:
        c = gen(s)
        sa = gen(s, L.Intervention(model, a.layer, scar_dirs, "ablate"))
        ra = gen(s, L.Intervention(model, a.layer, rand_dirs, "ablate"))
        pc, mc = analyze(c)
        psa, msa = analyze(sa)
        pra, mra = analyze(ra)
        d_scar, d_rand = 1 - cos(pc, psa), 1 - cos(pc, pra)
        results.append(dict(seed=s, scar_mass_control=mc, scar_mass_scarabl=msa,
                            scar_mass_randabl=mra, div_scar=d_scar, div_rand=d_rand,
                            control=c, scar_ablated=sa, random_ablated=ra))
        print(f"seed {s}: scar_mass ctrl={mc:.4f} scar-abl={msa:.4f} rand-abl={mra:.4f} "
              f"| div_scar={d_scar:.3f} div_rand={d_rand:.3f}", flush=True)

    agg = dict(n_seeds=len(results), scar_features=feats, random_features=rand_feats,
               mean_scar_mass_control=st.mean(r["scar_mass_control"] for r in results),
               mean_scar_mass_scarabl=st.mean(r["scar_mass_scarabl"] for r in results),
               mean_scar_mass_randabl=st.mean(r["scar_mass_randabl"] for r in results),
               mean_div_scar=st.mean(r["div_scar"] for r in results),
               mean_div_rand=st.mean(r["div_rand"] for r in results))
    json.dump(dict(aggregate=agg, seeds=results), open(a.out, "w"), indent=1)
    print("\nAGGREGATE:", json.dumps(agg, indent=1), flush=True)
    print("VERDICT: scar CAUSAL iff scar_mass_scarabl << control (suppressed) "
          "AND mean_div_scar > mean_div_rand (reorganises more than random).", flush=True)


if __name__ == "__main__":
    main()
