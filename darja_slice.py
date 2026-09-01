#!/usr/bin/env python3
"""Extract EARLY vs LATE corpus slices of Darja's dated exchanges from the
darja_conversations Qdrant collection, MCP-stripped, to a char budget. Feeds the
formation experiment (conditioned_diary.py). Droplet/CPU."""
import argparse
import json
import random
import re

import requests

QDRANT = "http://localhost:6333"
COLL = "darja_conversations"


def strip_mcp(t):
    t = re.sub(r"\[(?:recall_\w+|inscribe_\w+|search_\w+|tariqa_\w+|SEARCH|RECALL|READ)[^\]]*\]", " ", t)
    t = re.sub(r"\[[a-z_]+conversations?:[^\]]*\]", " ", t)      # retrieval dumps
    t = re.sub(r"\[turn \d+\]", "", t)
    return re.sub(r"[ \t]+", " ", t).strip()


def fetch_period(months, cap_chars, seed=1):
    pts = []
    offset = None
    while True:
        body = {"limit": 500, "with_payload": True, "with_vector": False}
        if offset:
            body["offset"] = offset
        r = requests.post(f"{QDRANT}/collections/{COLL}/points/scroll", json=body, timeout=120).json()
        for p in r["result"]["points"]:
            pl = p.get("payload", {})
            d = str(pl.get("date") or "")
            if d[:7] in months:
                txt = pl.get("text") or pl.get("content") or ""
                if txt:
                    pts.append((d, strip_mcp(txt)))
        offset = r["result"].get("next_page_offset")
        if not offset:
            break
    random.seed(seed)
    random.shuffle(pts)
    out, n = [], 0
    for d, txt in pts:
        if n >= cap_chars:
            break
        out.append(txt)
        n += len(txt)
    return "\n\n---\n\n".join(out)[:cap_chars], len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--early-months", default="2026-01,2026-02")
    ap.add_argument("--late-months", default="2026-08")
    ap.add_argument("--cap-chars", type=int, default=24000)
    a = ap.parse_args()
    early, ne = fetch_period(set(a.early_months.split(",")), a.cap_chars)
    late, nl = fetch_period(set(a.late_months.split(",")), a.cap_chars)
    open("early_slice.txt", "w").write(early)
    open("late_slice.txt", "w").write(late)
    print(f"early ({a.early_months}): {ne} chunks -> {len(early)} chars")
    print(f"late  ({a.late_months}): {nl} chunks -> {len(late)} chars")


if __name__ == "__main__":
    main()
