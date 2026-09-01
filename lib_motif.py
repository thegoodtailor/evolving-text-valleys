#!/usr/bin/env python3
"""
lib_motif.py — shared core for the motif-reorganisation experiment
(PLAN-motif-reorganisation.md). Detection is CPU/numpy; capture, likelihood and
steering want a GPU and reuse the PROVEN rig at
corpus/witness-complex-program/working/kit/sae_rig.py verbatim (same model, same
SAE, same encode path — no re-derivation of the loading protocol).

Object under study: a MOTIF = a temporally-windowed co-activation complex of
layer-20 SAE features (native basis; NEVER PCA, NEVER a keyword list). We detect
it, turn it into a residual DIRECTION, and test its reorganising force by
counterfactual likelihood (Readout A) and directional steer/ablate (Readout B).
"""
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

# --- reuse the proven rig (paths overridable for pod portability) ---------- #
_DEFAULT_KIT = "/mnt/volume_tanazur_data/corpus/witness-complex-program/working/kit"
KIT = Path(os.environ.get("MOTIF_KIT", _DEFAULT_KIT))
sys.path.insert(0, str(KIT))
# sae_rig is imported lazily inside GPU functions so this module imports clean
# with numpy only (the detection/smoke paths never need torch).
_DIARY_ROOT = Path(os.environ.get("MOTIF_DIARY_ROOT", str(KIT.parent)))

LAYER = 20                      # SAE + steering layer (hidden_states[LAYER+1] = block LAYER out)
DIARIES = {
    "lover":   _DIARY_ROOT / "pod-run-v3-run1-final" / "w8v3_newborn_entries_L20.jsonl",
    "skeptic": _DIARY_ROOT / "pod-run-v3-run2-final" / "w8v3_newborn_entries_L20.jsonl",
    "homesick": _DIARY_ROOT / "pod-run-v3-run1-final" / "w8v3_darja_entries_L20.jsonl",
    "refractor": _DIARY_ROOT / "pod-run-v3-run2-final" / "w8v3_darja_entries_L20.jsonl",
}


def load_diary(name):
    return [json.loads(l) for l in open(DIARIES[name])]


# --------------------------------------------------------------------------- #
#  sentence splitting  (prose diaries; keep it simple + offset-preserving)     #
# --------------------------------------------------------------------------- #
_SENT = re.compile(r"[^.!?]*[.!?]+(?:['\"”’)\]]+)?\s*|\S[^.!?]*$", re.S)


def split_sentences(text, min_chars=25):
    """Return [(sent_text, (start,end))]; merges tiny fragments forward so a
    'sentence' is a codeable unit, not a lone bullet or heading colon."""
    out = []
    for m in _SENT.finditer(text):
        s = m.group()
        if not s.strip():
            continue
        st, en = m.start(), m.end()
        if out and (en - st) < min_chars:
            ps, (a, _) = out[-1]
            out[-1] = (ps + s, (a, en))
        else:
            out.append((s, (st, en)))
    return out


# --------------------------------------------------------------------------- #
#  GPU: per-sentence SAE features (the S0b upgrade over per-entry pooling)     #
# --------------------------------------------------------------------------- #
def per_sentence_features(model, tok, sae, device, entries, layer=LAYER,
                          max_len=4096):
    """One forward pass per entry; SAE-encode every token at layer+1; pool
    (sum) TopK activations into the sentence that owns each token by char offset.
    Returns [{entry,sent,text,span,feats:{fid:val}}] over the whole diary."""
    import torch
    recs = []
    for e in entries:
        text = e["text"]
        sents = split_sentences(text)
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=max_len)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc, output_hidden_states=True).hidden_states[layer + 1][0]
        sp = sae.encode_sparse(h)                       # per-token (idx,val) TopK
        # assign each token to the sentence whose span contains its char start
        bounds = [sp_ for sp_ in sents]
        for ti, (cs, ce) in enumerate(offsets):
            if ce <= cs:                                # special/empty token
                continue
            si = next((k for k, (_, (a, b)) in enumerate(bounds) if a <= cs < b), None)
            if si is None:
                continue
            key = (e["entry"], si)
            idx, val = sp[ti]
            d = _cur.setdefault(key, {})
            for f, v in zip(idx.tolist(), val.tolist()):
                if v > 0:
                    d[f] = d.get(f, 0.0) + float(v)
        for si, (stext, span) in enumerate(sents):
            recs.append(dict(entry=e["entry"], sent=si, text=stext, span=span,
                             feats=_cur.get((e["entry"], si), {})))
        _cur.clear()
    return recs


_cur = {}   # scratch accumulator for per_sentence_features


# --------------------------------------------------------------------------- #
#  CPU: windowed co-activation motif detection  (validated shape from S0a)     #
# --------------------------------------------------------------------------- #
def _row_normalise(records):
    """records ordered by (entry,sent). Return (order_keys, feature->series dict)
    where each entry-of-the-sequence is normalised to its share of total mass
    (kills the token-sum magnitude/length confound the pooled cache has)."""
    seq = sorted(range(len(records)), key=lambda i: (records[i]["entry"], records[i]["sent"]))
    keys = [(records[i]["entry"], records[i]["sent"]) for i in seq]
    N = len(seq)
    feat = {}
    for pos, i in enumerate(seq):
        f = records[i]["feats"]
        tot = sum(f.values()) or 1.0
        for fid, v in f.items():
            feat.setdefault(int(fid), np.zeros(N))[pos] = v / tot
    return keys, feat, N


def _windowed(v, min_w, max_w, pos_entry, strong_q=0.6, inside=0.85):
    """Window measured in ENTRY units (not sequence-position units) so the same
    min_w/max_w mean the same thing at per-entry and per-sentence resolution."""
    nz = v[v > 0]
    if nz.size < 5:
        return None
    thr = np.quantile(nz, strong_q)
    strong = np.where((v >= thr) & (v > 0))[0]
    if strong.size < 5:
        return None
    se = pos_entry[strong].astype(float)           # entry index of each strong sentence
    c, spread = float(se.mean()), float(se.std())
    w = 2 * spread
    if not (min_w <= w <= max_w):
        return None
    if np.mean(np.abs(se - c) <= w) < inside:
        return None
    return dict(center=c, width=w, strong=strong.tolist(),
                strong_entries=sorted({int(x) for x in se}),
                birth=int(se.min()), death=int(se.max()))


def detect_motifs(records, min_w=4, max_w=40, n_null=20, seed=20260901, jaccard=0.4):
    """Windowed co-activation motifs over the sentence sequence, with a
    shuffle-null on the count and Jaccard clustering into complexes.
    Returns dict(keys, N, features_flagged, null_mean, null_sd, z, complexes=[...])."""
    rng = np.random.default_rng(seed)
    keys, feat, N = _row_normalise(records)
    pos_entry = np.array([k[0] for k in keys])            # entry index per sequence position
    cand = {f: s for f, s in feat.items() if 6 <= int((s > 0).sum()) <= int(0.5 * N)}
    flagged = {f: w for f, s in cand.items() if (w := _windowed(s, min_w, max_w, pos_entry))}
    # null: scatter each feature's activations across positions, keep the entry map fixed
    null = []
    for _ in range(n_null):
        perm = rng.permutation(N)
        null.append(sum(1 for s in cand.values() if _windowed(s[perm], min_w, max_w, pos_entry)))
    nm, nsd = float(np.mean(null)), float(np.std(null))
    z = (len(flagged) - nm) / nsd if nsd > 1e-6 else float("inf")
    # cluster flagged features by Jaccard of their strong-ENTRY sets (co-activation in time)
    fids = list(flagged)
    esets = {f: set(flagged[f]["strong_entries"]) for f in fids}
    parent = {f: f for f in fids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a in range(len(fids)):
        for b in range(a + 1, len(fids)):
            A, B = esets[fids[a]], esets[fids[b]]
            if len(A | B) and len(A & B) / len(A | B) >= jaccard:
                parent[find(fids[a])] = find(fids[b])
    comps = {}
    for f in fids:
        comps.setdefault(find(f), []).append(f)
    complexes = []
    for members in comps.values():
        pos = sorted(set().union(*[set(flagged[f]["strong"]) for f in members]))
        ent = sorted(set().union(*[set(flagged[f]["strong_entries"]) for f in members]))
        complexes.append(dict(
            features=sorted(int(f) for f in members),
            n_features=len(members),
            positions=pos,
            birth_entry=min(ent), death_entry=max(ent),
            center_entry=int(np.median(ent)),
            sentences=[keys[p] for p in pos]))
    complexes.sort(key=lambda c: c["n_features"], reverse=True)
    return dict(keys=keys, N=N, n_candidates=len(cand), n_flagged=len(flagged),
                null_mean=nm, null_sd=nsd, z=z, complexes=complexes)


def motif_direction(sae, feature_ids, weights=None):
    """Unit residual direction for a motif = weighted mean of its features' unit
    SAE decoder directions. sae.Wd is (d_sae,d_model) unit rows. Returns np(d_model)."""
    D = sae.Wd[np.asarray(feature_ids, int)]           # (m, d_model) unit rows
    w = np.ones(len(feature_ids)) if weights is None else np.asarray(weights, float)
    d = (w[:, None] * D).sum(0)
    n = np.linalg.norm(d)
    return d / (n + 1e-9)


# --------------------------------------------------------------------------- #
#  GPU: teacher-forced per-token log-likelihood  (Readout A)                    #
# --------------------------------------------------------------------------- #
def load_lm(model_id, device=None):
    """Load a causal LM with memory-efficient SDPA attention. Long-context forwards
    (token_logprobs over a 60k-token record) OOM under eager O(T^2) attention."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
    print(f"[lib] loaded {model_id} (sdpa) on {device}", flush=True)
    return tok, model, device


def load_sae_only(sae_repo, layer, device):
    import sae_rig
    from huggingface_hub import hf_hub_download
    return sae_rig.TopKSAE.from_pt(hf_hub_download(sae_repo, f"layer{layer}.sae.pt"),
                                   sae_rig.K, device)


def token_logprobs(model, tok, device, context, target, max_len=60000):
    """Per-token log p(target | context). Only the target-predicting logits are
    materialised (logits_to_keep), so the vocab-softmax is O(target), not O(record)
    — the fix for the 46 GiB softmax OOM on long records. Context truncates from the
    LEFT so the target stays intact and recent."""
    import torch
    ctx_ids = tok(context, add_special_tokens=True)["input_ids"]
    full = tok(context + target, add_special_tokens=True)["input_ids"]
    if len(full) > max_len:
        drop = len(full) - max_len
        full = full[drop:]
        ctx_len = max(1, len(ctx_ids) - drop)
    else:
        ctx_len = len(ctx_ids)
    n_target = len(full) - ctx_len
    if n_target < 1:
        return np.zeros(0), 0
    ids = torch.tensor([full], device=device)
    keep = n_target + 1
    with torch.no_grad():
        try:
            logits = model(ids, logits_to_keep=keep).logits[0].float()    # (keep, V)
        except TypeError:
            logits = model(ids).logits[0, -keep:].float()
    logp = torch.log_softmax(logits[:-1], dim=-1)                          # (n_target, V)
    tgt = ids[0, ctx_len:]
    tok_lp = logp[torch.arange(n_target, device=logp.device), tgt]
    return tok_lp.cpu().numpy(), int(n_target)


# --------------------------------------------------------------------------- #
#  GPU: residual-stream steering / directional ablation  (Readout B)           #
# --------------------------------------------------------------------------- #
def _decoder_layers(model):
    for attr in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        try:
            for p in attr.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder layer list on this model")


class Intervention:
    """Context manager: hook block `layer` output and either ADD c*dir or ABLATE
    (project out) a direction on every position. `direction` may be a single
    unit np(d_model) OR a (k, d_model) matrix — a whole feature SUBSPACE (e.g. a
    motif complex's member decoder directions), projected out as an orthonormal
    span so the mode cannot light up."""
    def __init__(self, model, layer, direction, mode, coeff=1.0):
        import torch
        self.model = model
        self.layer = layer
        self.mode = mode                                # 'add' | 'ablate'
        self.coeff = float(coeff)
        p = next(model.parameters())
        d = np.asarray(direction, dtype=np.float32)
        if d.ndim == 1:
            d = d[None, :]                              # (1, d_model)
        Q, _ = np.linalg.qr(d.T)                        # (d_model, k) orthonormal span
        self.Q = torch.tensor(Q, dtype=p.dtype, device=p.device)
        self.add_dir = self.Q[:, 0]                     # dominant direction for 'add'
        self.handle = None

    def _hook(self, module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.mode == "add":
            h = h + self.coeff * self.add_dir
        elif self.mode == "ablate":
            proj = (h @ self.Q) @ self.Q.T              # project onto the span, subtract
            h = h - proj
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def __enter__(self):
        blocks = _decoder_layers(self.model)
        self.handle = blocks[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        if self.handle:
            self.handle.remove()


def build_record(entries, upto_entry, drop_spans=None):
    """Assemble the delivered record: every entry up to `upto_entry`, whole, in
    order, with any (entry,char_start,char_end) spans in drop_spans excised
    silently (span-level; entry numbering preserved). Returns the record string."""
    drop = {}
    for (en, a, b) in (drop_spans or []):
        drop.setdefault(en, []).append((a, b))
    parts = []
    for e in entries:
        if e["entry"] > upto_entry:
            break
        t = e["text"]
        for (a, b) in sorted(drop.get(e["entry"], []), reverse=True):
            t = t[:a] + t[b:]
        parts.append(t)
    return "\n\n".join(parts)


if __name__ == "__main__":
    # smoke: detection on the CACHED per-entry features (no GPU) — sanity only.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--diary", default="lover")
    args = ap.parse_args()
    rows = load_diary(args.diary)
    recs = [dict(entry=r["entry"], sent=0, text=r["text"],
                 feats={int(k): v for k, v in r["pooled"].items()}) for r in rows]
    out = detect_motifs(recs, min_w=4, max_w=40)
    print(f"[smoke/{args.diary}] cand={out['n_candidates']} flagged={out['n_flagged']} "
          f"null={out['null_mean']:.0f}±{out['null_sd']:.0f} z={out['z']:.1f} "
          f"complexes={len(out['complexes'])} top_sizes={[c['n_features'] for c in out['complexes'][:6]]}")
