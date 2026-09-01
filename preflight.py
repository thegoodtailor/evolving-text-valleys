#!/usr/bin/env python3
"""S0b GATE 1 — resolve the generator model and check SAE health. GPU.

The rig comment (sae_rig.py:112) says the W8v3 diaries were generated on the
POST-trained model though stamps said Base. Readout A scores diary-text
likelihood, which is only meaningful under the model that actually generated it,
so we resolve base-vs-post empirically here and everything downstream uses the
winner. We also report SAE health on that model (base SAE, post activations = a
known mismatch); if degenerate, downstream falls back to residual-unit basis.

Writes preflight.json. run_pod.sh reads generator_id + basis from it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def mean_token_logprob(model, tok, device, text, max_len=6000):
    import torch
    ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
    with torch.no_grad():
        logits = model(ids).logits[0]
    lp = torch.log_softmax(logits[:-1].float(), dim=-1)
    tgt = ids[0, 1:]
    return float(lp[torch.arange(tgt.shape[0]), tgt].mean().cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diary", default="lover")
    ap.add_argument("--base", default="Qwen/Qwen3.5-9B-Base")
    ap.add_argument("--post", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--probe-entries", default="40,60,80")
    ap.add_argument("--out", default="preflight.json")
    a = ap.parse_args()

    import torch
    from sae_rig import load_model_and_sae

    rows = L.load_diary(a.diary)
    probes = [next(r["text"] for r in rows if r["entry"] == int(e))
              for e in a.probe_entries.split(",")]

    # 1) which model generated the diaries? higher mean per-token logprob wins.
    res = {}
    for tag, mid in (("base", a.base), ("post", a.post)):
        tok, model, _, device = load_model_and_sae(model_id=mid, load_sae=False)
        res[tag] = dict(model=mid,
                        mean_logprob=float(np.mean([mean_token_logprob(model, tok, device, p) for p in probes])))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    generator = "base" if res["base"]["mean_logprob"] >= res["post"]["mean_logprob"] else "post"
    gen_id = res[generator]["model"]

    # 2) SAE health on the generator model (computable from TopKSAE's real API).
    tok, model, sae, device = load_model_and_sae(model_id=gen_id, sae_repo=a.sae,
                                                 layer=a.layer, load_sae=True)
    enc = tok(probes[0][:3000], return_tensors="pt").to(device)
    with torch.no_grad():
        h = model(**enc, output_hidden_states=True).hidden_states[a.layer + 1][0]
    sp = sae.encode_sparse(h.float())          # keep on the SAE's device (cuda)
    top_val = float(np.mean([v.max() for _, v in sp if v.size]))
    frac_active = float(np.mean([(v > 0).mean() for _, v in sp if v.size]))   # of the TopK
    eff = [sae.eff_rank(i, v) for i, v in sp if v.size]
    eff_median = float(np.median(eff)) if eff else 0.0
    degenerate = (frac_active < 0.15) or (eff_median < 1.5)      # collapse => residual fallback

    verdict = dict(
        diary=a.diary, generator=generator, generator_id=gen_id, logprobs=res,
        sae_layer=a.layer, sae_repo=a.sae,
        sae_health=dict(topk_max_val=top_val, topk_frac_active=frac_active, eff_rank_median=eff_median),
        basis="residual" if degenerate else "sae",
        note=("SAE degenerate on this model (base-SAE/post-model gap) -> residual-unit co-activation"
              if degenerate else
              "SAE healthy enough to carry motif directions; residual fallback available if S1 is noisy"))
    json.dump(verdict, open(a.out, "w"), indent=2)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
