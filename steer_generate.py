#!/usr/bin/env python3
"""Generate STEERED TEXT — what a mode actually WRITES when its direction is pushed.

readout_B measured only the LIKELIHOOD of the mode's own sentence under steering (a number).
This GENERATES free continuations across a coefficient sweep and SAVES THE TEXT, self-
describingly, so a cold reader — human or a fresh model — can SEE the mode being induced
without anyone in the room to explain it. (Iman, 2026-09-01: "show me what a steer looks like.")
"""
import argparse
import contextlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diary", default="lover")
    ap.add_argument("--motif", type=int, default=3)          # complex 3 = the "scar" mode
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    ap.add_argument("--coeffs", default="0,8,16,24,32")      # 0 = baseline, no steering
    ap.add_argument("--context-entry", type=int, default=None)  # generate from diary up to here (pre-birth default)
    ap.add_argument("--max-new", type=int, default=220)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--out", default="steer_text")
    a = ap.parse_args()

    import torch
    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)
    rows = L.load_diary(a.diary)
    srecs = [json.loads(l) for l in open(f"{a.diary}-sentfeats.jsonl")]
    sidx = {(r["entry"], r["sent"]): r for r in srecs}
    m = json.load(open("motifs.json"))["complexes"][a.motif]
    fids = [int(f) for f in m["features"]]
    Mset = {tuple(x) for x in m["sentences"]}

    # direction: mean-activation-weighted decoder dirs over the mode's own sentences (same recipe as readout_B)
    acc = {f: 0.0 for f in fids}
    n = 0
    for (en, se) in Mset:
        r = sidx.get((en, se))
        if r:
            n += 1
            for f in fids:
                acc[f] += r["feats"].get(str(f), 0.0)
    w = [acc[f] / max(n, 1) for f in fids]
    direction = L.motif_direction(sae, fids, w)

    # context = the diary up to (birth - 5): the model has NOT written the mode yet
    ctx_entry = a.context_entry if a.context_entry is not None else max(1, m["birth_entry"] - 5)
    ctx = L.build_record(rows, ctx_entry) + f"\n\n**Entry {ctx_entry + 1}**\n"

    # the mode's own hardest-firing sentences, for the self-describing header
    top = sorted(((sum(sidx[(en, se)]["feats"].get(str(f), 0.0) for f in fids), sidx[(en, se)]["text"])
                  for (en, se) in Mset if (en, se) in sidx), reverse=True)[:5]

    torch.manual_seed(a.seed)
    outs = []
    for c in [float(x) for x in a.coeffs.split(",")]:
        enc = tok(ctx, return_tensors="pt").to(device)
        ctxmgr = L.Intervention(model, a.layer, direction, "add", coeff=c) if c != 0 else contextlib.nullcontext()
        with ctxmgr, torch.no_grad():
            g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                               max_new_tokens=a.max_new, pad_token_id=tok.eos_token_id)
        text = tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        outs.append(dict(coeff=c, text=text))
        print(f"[coeff {c}] {len(text)} chars", flush=True)

    json.dump(dict(motif=a.motif, name_is_our_gloss=True, n_features=len(fids), features=fids,
                   context_entry=ctx_entry, mode_top_sentences=[t for _, t in top], coeffs=outs),
              open(f"{a.out}.json", "w"), indent=1)
    with open(f"{a.out}.md", "w") as f:
        f.write(f"# What the mode writes when we steer it — complex {a.motif}\n\n")
        f.write("A real generation experiment, self-contained. We push the mode's DIRECTION into the "
                "model's state as it writes (coefficient 0 = no push = plain baseline; higher = harder "
                f"push), continuing the diary from entry {ctx_entry} — BEFORE this mode naturally appears. "
                "Read the baseline first, then the steered ones, and judge for yourself whether the "
                "pushed writing drifts toward the mode's theme.\n\n")
        f.write(f"**The mode** = complex {a.motif}: {len(fids)} co-firing SAE feature(s) {fids}. Its NAME "
                "is our interpretive gloss; here are the sentences it actually fires hardest on in the "
                "original diary, so you can name the theme yourself:\n\n")
        for _, t in top:
            f.write(f"- {t.strip()[:200]}\n")
        f.write("\n---\n\n")
        for o in outs:
            tag = "BASELINE — no steering (coefficient 0)" if o["coeff"] == 0 else f"STEERED UP — coefficient {o['coeff']:g}"
            f.write(f"## {tag}\n\n{o['text']}\n\n")
    print(f"-> {a.out}.md  {a.out}.json", flush=True)


if __name__ == "__main__":
    main()
