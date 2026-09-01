import os, time
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import snapshot_download, hf_hub_download

t0 = time.time()
def log(m): print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

MODELS = ["Qwen/Qwen3.5-9B-Base", "Qwen/Qwen3.5-9B"]
SAE = "Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100"
pat = ["*.safetensors", "*.json", "*.txt", "tokenizer*", "*.model"]
for mid in MODELS:
    try:
        log(f"fetch {mid}")
        snapshot_download(mid, allow_patterns=pat)
        log(f"  {mid} DONE")
    except Exception as e:
        log(f"  {mid} FAILED: {type(e).__name__}: {e}")
try:
    log(f"fetch SAE {SAE}")
    p = hf_hub_download(SAE, "layer20.sae.pt")
    log(f"  SAE DONE at {p}")
except Exception as e:
    log(f"  SAE FAILED: {type(e).__name__}: {e}")
log("DOWNLOAD_COMPLETE")
