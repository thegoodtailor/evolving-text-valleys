#!/usr/bin/env python3
"""S0b detect — windowed co-activation motifs at sentence resolution. CPU.
Reads <diary>-sentfeats.jsonl; writes motifs.json (machine) + motifs-readable.md
(so Iman can SEE what each complex is about before we spend on the readouts)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diary", default="lover")
    ap.add_argument("--feats", default=None)
    ap.add_argument("--min-w", type=int, default=4)
    ap.add_argument("--max-w", type=int, default=40)
    a = ap.parse_args()

    feats = a.feats or f"{a.diary}-sentfeats.jsonl"
    recs = [json.loads(l) for l in open(feats)]
    for r in recs:
        r["feats"] = {int(k): v for k, v in r["feats"].items()}
    out = L.detect_motifs(recs, min_w=a.min_w, max_w=a.max_w)
    txt = {(r["entry"], r["sent"]): r["text"] for r in recs}

    dump = {k: v for k, v in out.items() if k != "keys"}
    json.dump(dump, open("motifs.json", "w"), indent=2, default=str)
    with open("motifs-readable.md", "w") as f:
        f.write(f"# Motifs — {a.diary}\n\n")
        f.write(f"sentences={out['N']}  candidates={out['n_candidates']}  "
                f"flagged={out['n_flagged']}  null={out['null_mean']:.0f}±{out['null_sd']:.0f}  "
                f"**z={out['z']:.1f}**  complexes={len(out['complexes'])}\n\n")
        f.write("A complex is a candidate motif: features that co-activate in a bounded window. "
                "The biggest one is usually the back-half drift (rung 2); the interesting ones are "
                "mid-size complexes with a tight birth->death window that *read* as a distinction.\n\n")
        for i, c in enumerate(out["complexes"][:15]):
            f.write(f"## complex {i}: {c['n_features']} feats · born e{c['birth_entry']} "
                    f"· centre e{c['center_entry']} · dies e{c['death_entry']}\n")
            for (en, se) in c["sentences"][:5]:
                f.write(f"- e{en}.s{se}: {txt.get((en, se), '').strip()[:170]}\n")
            f.write("\n")
    print(f"[detect] z={out['z']:.1f}  complexes={len(out['complexes'])}  "
          f"top_sizes={[c['n_features'] for c in out['complexes'][:6]]} -> motifs.json, motifs-readable.md")


if __name__ == "__main__":
    main()
