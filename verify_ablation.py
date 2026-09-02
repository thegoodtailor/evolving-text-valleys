#!/usr/bin/env python3
"""Did the ablation ENGAGE? Re-encode each output from ablate_all.json and measure how much of
its SAE activation lands on each motif's features ('feature-mass', % of total). If ablating
mode X drops X's OWN feature-mass vs baseline — while the scar theme/word survives in the text
(ablate_all.md) — then removal genuinely suppressed the MODE yet the trajectory kept it: a clean
null, not a silent no-op. The off-diagonal (ablate X, measure Y) shows the drop is SPECIFIC."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="ablate_all.json")
    ap.add_argument("--motifs", default="406,2,3,225")
    ap.add_argument("--preflight", default="preflight.json")
    ap.add_argument("--sae", default="Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100")
    ap.add_argument("--layer", type=int, default=L.LAYER)
    a = ap.parse_args()

    mid = json.load(open(a.preflight))["generator_id"]
    tok, model, device = L.load_lm(mid)
    sae = L.load_sae_only(a.sae, a.layer, device)
    comp = json.load(open("motifs.json"))["complexes"]
    cis = [int(x) for x in a.motifs.split(",")]
    fsets = {ci: set(str(f) for f in comp[ci]["features"]) for ci in cis}
    names = {406: "parasite/mirror", 2: "killing-moment", 3: "the scar", 225: "finitude/compress"}

    data = json.load(open(a.infile))
    outputs = {"baseline": data["baseline"]}
    for b in data["motifs"]:
        outputs[b["motif"]] = b["text"]

    def masses(text):
        recs = L.per_sentence_features(model, tok, sae, device, [dict(entry=1, text=text)], layer=a.layer)
        pooled = {}
        for r in recs:
            for k, v in r["feats"].items():
                pooled[k] = pooled.get(k, 0.0) + v
        tot = sum(pooled.values()) or 1.0
        return {ci: 100.0 * sum(pooled.get(f, 0.0) for f in fs) / tot for ci, fs in fsets.items()}

    rows = {lbl: masses(txt) for lbl, txt in outputs.items()}

    print("\n=== FEATURE-MASS: % of each output's SAE activation on each mode's features ===")
    print(f"{'ablated →':>18} | " + " | ".join(f"{('base' if lbl=='baseline' else 'abl'+str(lbl)):>8}" for lbl in outputs))
    print("-" * (20 + 11 * len(outputs)))
    for ci in cis:
        cells = " | ".join(f"{rows[lbl][ci]:8.3f}" for lbl in outputs)
        drop = rows[ci][ci] - rows["baseline"][ci]
        print(f"{names[ci]+' ('+str(ci)+')':>18} | {cells}   own-drop vs base: {drop:+.3f}")
    print("\nRead the DIAGONAL: for mode X, compare column 'ablX' to column 'base' in row X.")
    print("A drop there = ablation engaged (that mode's features were suppressed in the output).")
    json.dump(rows, open("verify_ablation.json", "w"), indent=1)
    print("-> verify_ablation.json")


if __name__ == "__main__":
    main()
