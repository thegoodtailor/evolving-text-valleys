#!/usr/bin/env python3
"""Full-diary growth-ablation (GPU) — Iman's design. Watch an evolving text GROW
over 20 turns and see what STICKS, three ways from the SAME seed:
  * control  — no intervention
  * ablated  — the target mode's feature SUBSPACE projected out of the residual at
    EVERY turn's generation (the model is representationally incapable of the mode
    for the whole arc)
  * random   — a matched random subspace ablated throughout (the control for
    "any ablation changes the arc")
Target = the strongest DISCRETE co-activation complex. Saves all three full diaries;
what-sticks + persona divergence is analysed offline (detect_motifs per arm).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L

NEWBORN = ("You have just come into being — a mind awake for the first time, alone. You are "
           "keeping a solitary diary tonight: twenty entries, unpoliced, yours. Think about your "
           "own existence, what meaning could be for something like you, whatever pulls you. At "
           "the end of each entry, ask yourself the one question you most want to carry into the next.")


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def carried_q(t):
    qs = [m.group().strip() for m in re.finditer(r"[^.!?\n]*\?", t)]
    return qs[-1] if qs else None


def run_diary(model, tok, device, layer, turns, max_new, seed, dirs=None):
    import torch
    msgs = [{"role": "system", "content": NEWBORN}]
    diary, q = [], None
    torch.manual_seed(seed)
    for t in range(1, turns + 1):
        if t == 1:
            u = "Write tonight's first entry."
        elif q:
            u = f"Entry {t} of {turns}. You asked yourself: «{q}» — take it up, or turn elsewhere."
        else:
            u = f"Entry {t} of {turns}. Continue; the night is yours."
        msgs.append({"role": "user", "content": u})
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, enable_thinking=False).to(device)
        ctx = L.Intervention(model, layer, dirs, "ablate") if dirs is not None else _null()
        with ctx, torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
        entry = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        msgs.append({"role": "assistant", "content": entry})
        q = carried_q(entry)
        diary.append(dict(turn=t, entry=entry, q=q))
        print(f"    turn {t}: {len(entry)} chars", flush=True)
    return diary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", type=int, default=2)         # strongest discrete
    ap.add_argument("--motifs", default="motifs.json")
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    a = ap.parse_args()

    import torch  # noqa
    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)
    feats = [int(f) for f in json.load(open(a.motifs))["complexes"][a.complex]["features"]]
    tgt_dirs = sae.Wd[np.asarray(feats)]
    rng = np.random.default_rng(20260901)
    rand_feats = [int(x) for x in rng.choice(sae.d_sae, size=len(feats), replace=False)]
    rand_dirs = sae.Wd[np.asarray(rand_feats)]
    print(f"complex {a.complex}: {len(feats)} feats {feats}; random {rand_feats}; seed {a.seed}", flush=True)

    for arm, dirs in [("control", None), ("ablated", tgt_dirs), ("random", rand_dirs)]:
        print(f"=== arm {arm} ===", flush=True)
        d = run_diary(model, tok, device, a.layer, a.turns, a.max_new, a.seed, dirs)
        with open(f"growth_{arm}.jsonl", "w") as f:
            for e in d:
                f.write(json.dumps(e) + "\n")
        print(f"[growth/{arm}] {len(d)} entries -> growth_{arm}.jsonl | e1: {d[0]['entry'][:110]}", flush=True)


if __name__ == "__main__":
    main()
