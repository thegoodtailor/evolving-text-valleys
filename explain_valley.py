#!/usr/bin/env python3
"""Print a valley/motif complex IN FULL (for the write-up): its member SAE
features, and EVERY sentence where they co-activate, in context — so a reader sees
why it earns its name, and whether the complex is LEXICAL (the naming word is
present) or SEMANTIC/TONAL (it fires on other words/register too — the Phanes
point: a mode, not a figure). CPU, offline."""
import argparse
import json
import re


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motifs", default="motifs.json")
    ap.add_argument("--feats", default="lover-sentfeats.jsonl")
    ap.add_argument("--complex", type=int, default=3)
    ap.add_argument("--name-words", default="scar,wound,fracture,heal,bone,tissue,tourniquet,limb")
    a = ap.parse_args()

    m = json.load(open(a.motifs))["complexes"][a.complex]
    txt = {}
    for l in open(a.feats):
        r = json.loads(l)
        txt[(r["entry"], r["sent"])] = r["text"]
    words = [w.strip() for w in a.name_words.split(",") if w.strip()]
    pat = re.compile(r"\b(?:" + "|".join(words) + r")\w*", re.I)

    sents = sorted(tuple(s) for s in m["sentences"])
    print(f"COMPLEX {a.complex}: {m['n_features']} features | born e{m['birth_entry']} "
          f"center e{m['center_entry']} dies e{m['death_entry']}")
    print(f"member SAE features: {m['features']}")
    print(f"lit in {len(sents)} sentences  (name-words: {a.name_words})\n")
    hit = 0
    for (en, se) in sents:
        t = (txt.get((en, se), "")).replace("\n", " ").strip()
        has = bool(pat.search(t))
        hit += has
        print(f"  {'[LEX]' if has else '[TONE]'} e{en}.s{se}: {t[:220]}")
    n = len(sents)
    print(f"\nname-word present in {hit}/{n} sentences ({100*hit//max(n,1)}%).")
    print(f"the other {n-hit} are the complex firing on the MODE, not the word "
          f"-> {'LEXICAL' if hit==n else ('SEMANTIC/TONAL' if hit < n*0.8 else 'MOSTLY-LEXICAL')} valley.")


if __name__ == "__main__":
    main()
