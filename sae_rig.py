#!/usr/bin/env python3
"""
sae_rig.py — shared model/SAE/generation/fingerprint rig for the witness-complex
program. Lifted VERBATIM (where it works) from the proven ICRA-16 sense-cloud
scripts: geometry-of-sense/experiments/cloud_fingerprint.py, cloud_persona.py,
cloud_kitab.py. Same model, same SAE, same bare-prefix / T=1.0 ancestral-sampling
protocol, same TopK-SAE decode. Every experiment in the kit imports from here so
the loading/sampling path is IDENTICAL to the published rig.

Protocol (ICRA-16, sense-cloud.tex methods §):
  MODEL      Qwen3.5-9B-Base (base model — completions, NOT chat)
  SAE        Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100, residual, layer 20, TopK=100
  sampling   do_sample, temperature=1.0, top_p=1.0 (pure ancestral)
  prefix     bare (R8-clean) — no chat template, no instruction wrapper
  fingerprint  hidden_states[layer+1] over the GENERATED tokens only, TopK-SAE encode

CPU note: everything here degrades to CPU if no CUDA; the SAE decode + all
topology is numpy and runs on this box. Model download/forward is the only part
that wants a GPU — the experiments' --smoke paths never touch it.
"""
import argparse
import datetime
import json
import shutil
from pathlib import Path

import numpy as np

# torch / transformers / hf_hub are imported lazily inside load_model_and_sae so
# that witness_complex.py and the --smoke paths import clean with numpy only.

MODEL = "Qwen/Qwen3.5-9B-Base"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
K = 100
SEED = 11235


# --------------------------------------------------------------------------- #
#  TopK SAE  (verbatim from cloud_fingerprint.py)                             #
# --------------------------------------------------------------------------- #
class TopKSAE:
    def __init__(self, W_enc, b_enc, W_dec, k, device):
        import torch  # noqa: F401  (kept local so numpy-only imports stay clean)
        self.W_enc = W_enc.to(device)
        self.b_enc = b_enc.to(device)
        self.k = k
        self.d_sae, self.d_model = W_enc.shape
        Wd = W_dec.float()
        if Wd.shape[0] == self.d_model and Wd.shape[1] == self.d_sae:
            Wd = Wd.T                                     # (d_model,d_sae) -> (d_sae,d_model)
        Wd = Wd / (Wd.norm(dim=-1, keepdim=True) + 1e-9)  # unit decoder direction per feature
        self.Wd = Wd.cpu().numpy()

    @classmethod
    def from_pt(cls, path, k, device, dtype=None):
        import torch
        if dtype is None:
            dtype = torch.bfloat16
        sd = torch.load(path, map_location="cpu")
        c = lambda t: t.to(dtype)
        return cls(c(sd["W_enc"]), c(sd["b_enc"]), c(sd["W_dec"]), k, device)

    def encode_sparse(self, x):
        """x: (T, d_model) -> list of (idx_array int32, val_array float32) TopK per row."""
        import torch
        with torch.no_grad():
            x = x.to(self.W_enc.dtype)
            pre = x @ self.W_enc.T + self.b_enc
            vals, idx = pre.topk(self.k, dim=-1)
            vals = vals.relu().float().cpu().numpy()
            idx = idx.cpu().numpy()
        return [(idx[i].astype(np.int32), vals[i].astype(np.float32)) for i in range(idx.shape[0])]

    def eff_rank(self, idx, val):
        """superposition breadth: effective rank of the active features' unit decoder
        directions, activation-weighted. 1 = collinear (one reading); high = many
        independent meaning-directions co-active. (ICRA-16 'breadth')."""
        v = np.asarray(val, float)
        idx = np.asarray(idx)
        m = v > 0
        v = v[m]
        idx = idx[m]
        if idx.size < 2:
            return float(idx.size)
        D = self.Wd[idx]                    # (k, d_model) unit rows
        W = v[:, None] * D
        G = W @ W.T
        lam = np.linalg.eigvalsh(G)
        lam = lam[lam > 1e-12]
        if lam.size == 0:
            return 0.0
        return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))


# --------------------------------------------------------------------------- #
#  model + SAE loading                                                         #
# --------------------------------------------------------------------------- #
def load_model_and_sae(layer=20, model_id=MODEL, sae_repo=SAE_REPO, k=K,
                       load_sae=True, device=None):
    """Returns (tok, model, sae, device). sae is None if load_sae=False (calib)."""
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).to(device).eval()
    # record what was ACTUALLY loaded so rig_meta stamps the truth — the W8 v3
    # runs generated on the post-trained model while every stamped JSON said
    # Base, because rig_meta emitted the module constant (manifest §7).
    global _LOADED_MODEL_ID
    _LOADED_MODEL_ID = model_id
    print(f"[rig] loaded {model_id} on {device}", flush=True)
    sae = None
    if load_sae:
        sae = TopKSAE.from_pt(hf_hub_download(sae_repo, f"layer{layer}.sae.pt"), k, device)
        print(f"[rig] loaded SAE {sae_repo} layer{layer} (d_sae={sae.d_sae}, d_model={sae.d_model})",
              flush=True)
    return tok, model, sae, device


def make_generate(model, tok, device, max_new=28, batch=8, temperature=1.0,
                  top_p=1.0, max_prefix_len=1400):
    """Returns generate(prefix, n) -> list[str] of n continuations (prefix stripped).
    Bare prefix, ancestral sampling — the R8-clean protocol."""
    import torch

    def generate(prefix, n):
        conts = []
        while len(conts) < n:
            b = min(batch, n - len(conts))
            enc = tok([prefix] * b, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_prefix_len).to(device)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=temperature,
                                     top_p=top_p, max_new_tokens=max_new,
                                     pad_token_id=tok.pad_token_id)
            plen = enc["input_ids"].shape[1]
            for g in gen:
                conts.append(tok.decode(g[plen:], skip_special_tokens=True))
            print(f"    {len(conts)}/{n}", flush=True)
        return conts[:n]

    return generate


def make_fingerprints(model, tok, sae, layer, device, max_len=1500, max_prefix_len=1400):
    """Returns fingerprints(prefix, cont) -> (list[(idx,val)] per generated token, tokens).
    Encodes hidden_states[layer+1] over the GENERATED span only (ICRA-16)."""
    import torch

    def fingerprints(prefix, cont):
        plen = tok(prefix, return_tensors="pt", truncation=True,
                   max_length=max_prefix_len)["input_ids"].shape[1]
        e = tok(prefix + cont, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        ids = e["input_ids"][0].tolist()
        with torch.no_grad():
            h = model(**e, output_hidden_states=True).hidden_states[layer + 1][0]
        gen_h = h[plen:]
        if gen_h.shape[0] == 0:
            return [], []
        sp = sae.encode_sparse(gen_h)
        toks = [tok.decode([tid]) for tid in ids[plen:]]
        return sp, toks

    return fingerprints


# --------------------------------------------------------------------------- #
#  small shared utilities (verbatim from the sense-cloud scripts)             #
# --------------------------------------------------------------------------- #
def archive_then_write(path, obj):
    """Iman's never-delete law: back up any file we are about to overwrite."""
    path = Path(path)
    if path.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, f"{path}.bak-{stamp}")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def open_verbatim_log(path):
    """Open a JSONL sink for verbatim per-stage logging (production discipline).
    Backs up an existing file first, returns the open handle."""
    path = Path(path)
    if path.exists():
        shutil.copy2(path, f"{path}.bak-{datetime.datetime.now():%Y%m%d-%H%M%S}")
    return open(path, "w")


def participation_ratio(vals):
    v = np.asarray(vals, float)
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    return float((v.sum() ** 2) / (np.square(v).sum() + 1e-12))


def sparse_cos(a_idx, a_val, b_idx, b_val):
    da = {int(i): float(x) for i, x in zip(a_idx, a_val) if x > 0}
    db = {int(i): float(x) for i, x in zip(b_idx, b_val) if x > 0}
    if not da or not db:
        return 0.0
    dot = sum(da[i] * db[i] for i in da.keys() & db.keys())
    na = np.sqrt(sum(x * x for x in da.values()))
    nb = np.sqrt(sum(x * x for x in db.values()))
    return float(dot / (na * nb + 1e-12))


def sparse_jaccard(a_idx, a_val, b_idx, b_val):
    sa = {int(i) for i, x in zip(a_idx, a_val) if x > 0}
    sb = {int(i) for i, x in zip(b_idx, b_val) if x > 0}
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb) / (len(sa | sb) + 1e-12))


def pi0_sweep(D, eps_grid):
    """single-linkage component count at each eps on distance matrix D."""
    n = D.shape[0]
    out = []
    for eps in eps_grid:
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i in range(n):
            for j in range(i + 1, n):
                if D[i, j] <= eps:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj
        out.append(len(set(find(i) for i in range(n))))
    return out


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


_LOADED_MODEL_ID = None   # set by load_model_and_sae; None until a model loads


def rig_meta(layer, **extra):
    d = {"model": _LOADED_MODEL_ID or MODEL,
         "model_is_loaded_id": _LOADED_MODEL_ID is not None,
         "sae_repo": SAE_REPO, "layer": layer, "k": K, "seed": SEED,
         "timestamp_utc": now_utc()}
    d.update(extra)
    return d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="sanity: print rig constants")
    ap.parse_args()
    print(f"MODEL={MODEL}\nSAE_REPO={SAE_REPO}\nK={K}\nSEED={SEED}")
