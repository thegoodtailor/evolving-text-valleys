#!/usr/bin/env python3
"""Per-arm windowed co-activation z for the CLEAN conditioned diaries — WITHOUT
clobbering the canonical motifs.json. CPU-only (detect_motifs is pure numpy)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L

for arm in ["early", "late", "none"]:
    feats = f"diary_{arm}_clean-sentfeats.jsonl"
    recs = [json.loads(l) for l in open(feats)]
    for r in recs:
        r["feats"] = {int(k): v for k, v in r["feats"].items()}
    out = L.detect_motifs(recs)
    print(f"{arm}: N={out['N']} flagged={out['n_flagged']} "
          f"null={out['null_mean']:.0f}±{out['null_sd']:.0f} z={out['z']:.2f} "
          f"complexes={len(out['complexes'])} top={[c['n_features'] for c in out['complexes'][:5]]}",
          flush=True)
