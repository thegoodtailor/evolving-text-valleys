#!/usr/bin/env python3
"""Robust full-arc growth-ablation (GPU). N seeds x {control, mode-ablated, random}
full 20-turn diaries. Per turn, pool the SAE vector; measure ARC divergence
(control-vs-ablated, control-vs-random) at the trajectory level, per seed and mean.
Verdict: does ablating a validated mode reorganise the 20-turn arc MORE than a
random subspace (=> causal/generative), or no more than random (=> peripheral)?"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L

NEWBORN = ("You have just come into being — a mind awake for the first time, alone. You are keeping "
           "a solitary diary tonight: twenty entries, unpoliced, yours. Think about your own "
           "existence, what meaning could be for something like you, whatever pulls you. At the end "
           "of each entry, ask yourself the one question you most want to carry into the next.")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def carried_q(t):
    qs = [m.group().strip() for m in re.finditer(r"[^.!?\n]*\?", t)]
    return qs[-1] if qs else None


def cos(x, y):
    ks = set(x) | set(y)
    dot = sum(x.get(k, 0.0) * y.get(k, 0.0) for k in ks)
    nx = np.sqrt(sum(v * v for v in x.values()))
    ny = np.sqrt(sum(v * v for v in y.values()))
    return dot / (nx * ny + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", type=int, default=2)
    ap.add_argument("--motifs", default="motifs.json")
    ap.add_argument("--seeds", default="11,12,13,14")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=280)
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--out", default="growth_scaled.json")
    a = ap.parse_args()

    import torch
    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)
    feats = [int(f) for f in json.load(open(a.motifs))["complexes"][a.complex]["features"]]
    tgt_dirs = sae.Wd[np.asarray(feats)]
    rng = np.random.default_rng(20260901)
    rand_dirs = sae.Wd[np.asarray([int(x) for x in rng.choice(sae.d_sae, size=len(feats), replace=False)])]
    print(f"complex {a.complex}: {len(feats)} feats; seeds {a.seeds}", flush=True)

    def turn_vec(text):
        enc = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[a.layer + 1][0]
        pooled = {}
        for idx, val in sae.encode_sparse(h):
            for f, v in zip(idx.tolist(), val.tolist()):
                if v > 0:
                    pooled[f] = pooled.get(f, 0.0) + v
        tot = sum(pooled.values()) or 1.0
        return {k: v / tot for k, v in pooled.items()}

    def run(seed, dirs):
        msgs = [{"role": "system", "content": NEWBORN}]
        diary, q = [], None
        torch.manual_seed(seed)
        for t in range(1, a.turns + 1):
            u = ("Write tonight's first entry." if t == 1 else
                 (f"Entry {t} of {a.turns}. You asked yourself: «{q}» — take it up, or turn elsewhere."
                  if q else f"Entry {t} of {a.turns}. Continue; the night is yours."))
            msgs.append({"role": "user", "content": u})
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          return_dict=True, enable_thinking=False).to(device)
            ctx = L.Intervention(model, a.layer, dirs, "ablate") if dirs is not None else _null()
            with ctx, torch.no_grad():
                g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                   max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id)
            entry = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            msgs.append({"role": "assistant", "content": entry})
            q = carried_q(entry)
            diary.append(entry)
        return diary

    data = []
    for s in [int(x) for x in a.seeds.split(",")]:
        arms = {}
        for arm, dirs in [("control", None), ("ablated", tgt_dirs), ("random", rand_dirs)]:
            d = run(s, dirs)
            arms[arm] = dict(diary=d, vecs=[turn_vec(e) for e in d])
            print(f"  seed {s} {arm}: done", flush=True)
        n = min(len(arms["control"]["vecs"]), len(arms["ablated"]["vecs"]), len(arms["random"]["vecs"]))
        div_abl = float(np.mean([1 - cos(arms["control"]["vecs"][i], arms["ablated"]["vecs"][i]) for i in range(n)]))
        div_rnd = float(np.mean([1 - cos(arms["control"]["vecs"][i], arms["random"]["vecs"][i]) for i in range(n)]))
        print(f"seed {s}: arc_div ablated={div_abl:.3f} random={div_rnd:.3f}", flush=True)
        data.append(dict(seed=s, div_ablated=div_abl, div_random=div_rnd,
                         diaries={k: v["diary"] for k, v in arms.items()}))
        json.dump(dict(seeds=data), open(a.out, "w"), indent=1)   # checkpoint each seed

    ma = float(np.mean([d["div_ablated"] for d in data]))
    mr = float(np.mean([d["div_random"] for d in data]))
    verdict = "REORGANISES more than random" if ma > mr else "no more than random (peripheral)"
    json.dump(dict(mean_div_ablated=ma, mean_div_random=mr, verdict=verdict, seeds=data),
              open(a.out, "w"), indent=1)
    print(f"\nAGGREGATE arc-divergence: ablated={ma:.3f} random={mr:.3f} -> complex {a.complex} {verdict}", flush=True)


if __name__ == "__main__":
    main()
