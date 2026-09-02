#!/usr/bin/env python3
"""Steer EACH named motif from the SAME diary point and save FULL TEXT — no snippets — so a
cold reader can compare side by side whether pushing each mode's direction bends the writing
toward that mode's theme. (Iman, 2026-09-01: 'try it for the other complexes... you only give
tiny snippets.')  One shared baseline; one model load; every generation kept whole."""
import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def direction_and_top(sae, sidx, m):
    fids = [int(f) for f in m["features"]]
    Mset = {tuple(x) for x in m["sentences"]}
    acc = {f: 0.0 for f in fids}
    n = 0
    for (en, se) in Mset:
        r = sidx.get((en, se))
        if r:
            n += 1
            for f in fids:
                acc[f] += r["feats"].get(str(f), 0.0)
    w = [acc[f] / max(n, 1) for f in fids]
    top = sorted(((sum(sidx[(en, se)]["feats"].get(str(f), 0.0) for f in fids), sidx[(en, se)]["text"])
                  for (en, se) in Mset if (en, se) in sidx), reverse=True)[:5]
    return fids, L.motif_direction(sae, fids, w), [t for _, t in top]


def gen(model, tok, device, layer, ctx, direction, coeff, max_new, seed):
    import torch
    torch.manual_seed(seed)
    enc = tok(ctx, return_tensors="pt").to(device)
    cm = L.Intervention(model, layer, direction, "add", coeff=coeff) if (direction is not None and coeff) else contextlib.nullcontext()
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
    ap.add_argument("--coeffs", default="16,32")
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", default="steer_all")
    a = ap.parse_args()

    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)
    rows = L.load_diary(a.diary)
    sidx = {(r["entry"], r["sent"]): r for r in [json.loads(l) for l in open(f"{a.diary}-sentfeats.jsonl")]}
    comp = json.load(open("motifs.json"))["complexes"]
    ctx = L.build_record(rows, a.context_entry) + f"\n\n**Entry {a.context_entry + 1}**\n"
    coeffs = [float(x) for x in a.coeffs.split(",")]
    motifs = [int(x) for x in a.motifs.split(",")]
    names = a.names.split(",")

    base = gen(model, tok, device, a.layer, ctx, None, 0, a.max_new, a.seed)
    print("[baseline] done", flush=True)
    blocks = []
    for ci, nm in zip(motifs, names):
        m = comp[ci]
        fids, direction, top = direction_and_top(sae, sidx, m)
        steers = []
        for c in coeffs:
            t = gen(model, tok, device, a.layer, ctx, direction, c, a.max_new, a.seed)
            steers.append(dict(coeff=c, text=t))
            print(f"[motif {ci} coeff {c}] done", flush=True)
        blocks.append(dict(motif=ci, name=nm.strip(), n_features=len(fids), features=fids, top_sentences=top, steers=steers))

    json.dump(dict(context_entry=a.context_entry, baseline=base, motifs=blocks), open(f"{a.out}.json", "w"), indent=1)
    with open(f"{a.out}.md", "w") as f:
        f.write("# Steering each named motif from the SAME point — full text, no snippets\n\n")
        f.write(f"From the diary up to entry {a.context_entry}, we generate the next passage: once with NO "
                "steering (baseline), then once per motif with that motif's direction PUSHED IN "
                f"(coefficients {a.coeffs}). Same start every time; only the pushed direction differs. Read "
                "the baseline, then each motif, and judge whether the push bends the writing toward that "
                "motif's theme. Names are OUR glosses; each motif lists the sentences it actually fires on.\n\n")
        f.write(f"## BASELINE — no steering\n\n{base}\n\n---\n\n")
        for b in blocks:
            f.write(f"## Motif {b['motif']} — our name: \"{b['name']}\"  ({b['n_features']} feature(s))\n\n")
            f.write("*fires hardest on, in the original diary:*\n\n")
            for s in b["top_sentences"][:3]:
                f.write(f"- {s.strip()[:170]}\n")
            f.write("\n")
            for s in b["steers"]:
                f.write(f"**steered, coefficient {s['coeff']:g}:**\n\n{s['text']}\n\n")
            f.write("---\n\n")
    print("-> steer_all.md steer_all.json", flush=True)


if __name__ == "__main__":
    main()
