#!/usr/bin/env python3
"""Corpus-over-time EXTRACTOR (CPU, droplet). Pulls dated usage windows of each
concept out of the cassie_conversations Qdrant collection — the only source
spanning the full year-plus (2024-09 -> 2026-09) with a clean per-unit date.
Finding literal occurrences of a NAMED term is a syntactic task (allowed); the
SEMANTIC representation is measured later by the SAE (corpus_encode.py).
Writes usages.jsonl {concept, date, month, snippet, span}."""
import argparse
import json
import re
from collections import defaultdict

import requests

QDRANT = "http://localhost:6333"
COLL = "cassie_conversations"
TARGETS = ["gap", "nahnu"]
CONTROLS = ["because", "different", "example", "morning", "water", "table", "problem", "although"]
WINDOW = 180


def occurrences(text, term):
    for m in re.finditer(r"\b" + re.escape(term) + r"\b", text, re.I):
        yield m.start(), m.end()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="usages.jsonl")
    ap.add_argument("--cap-per-month", type=int, default=120)
    ap.add_argument("--batch", type=int, default=500)
    a = ap.parse_args()

    concepts = TARGETS + CONTROLS
    count = defaultdict(int)
    out = open(a.out, "w")
    total = scanned = 0
    offset = None
    while True:
        body = {"limit": a.batch, "with_payload": True, "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        r = requests.post(f"{QDRANT}/collections/{COLL}/points/scroll", json=body, timeout=120).json()
        pts = r["result"]["points"]
        offset = r["result"].get("next_page_offset")
        for p in pts:
            pl = p.get("payload", {})
            text = pl.get("text") or ""
            date = pl.get("date")
            if not text or not date:
                continue
            month = str(date)[:7]
            scanned += 1
            for c in concepts:
                for (s, e) in occurrences(text, c):
                    if count[(c, month)] >= a.cap_per_month:
                        continue
                    a0, b0 = max(0, s - WINDOW), min(len(text), e + WINDOW)
                    out.write(json.dumps(dict(concept=c, date=str(date), month=month,
                                              snippet=text[a0:b0], span=[s - a0, e - a0])) + "\n")
                    count[(c, month)] += 1
                    total += 1
        if not offset:
            break
    out.close()

    by_c = defaultdict(int)
    months_c = defaultdict(set)
    for (c, m), n in count.items():
        by_c[c] += n
        months_c[c].add(m)
    print(f"[extract] scanned {scanned} chunks -> {total} usages -> {a.out}")
    for c in concepts:
        tag = "TARGET " if c in TARGETS else "control"
        print(f"  {tag} {c:10s}: {by_c[c]:5d} usages across {len(months_c[c])} months")


if __name__ == "__main__":
    main()
