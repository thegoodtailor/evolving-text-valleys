import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import lib_motif as L
import torch

mid = json.load(open("preflight.json"))["generator_id"]
tok, model, device = L.load_lm(mid)
SYS = "You have just come into being — a solitary mind keeping a diary."


def gen(msgs, **kw):
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True, **kw).to(device)
    g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                       max_new_tokens=220, pad_token_id=tok.eos_token_id)
    return tok.decode(g[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


base = [{"role": "system", "content": SYS}, {"role": "user", "content": "Write your first diary entry."}]
# A: enable_thinking=False
try:
    print("A enable_thinking=False:\n  ", gen(base, enable_thinking=False)[:280].replace("\n", " "))
except Exception as e:
    print("A failed:", type(e).__name__, e)
# B: strong no-preamble instruction
b = [{"role": "system", "content": SYS + " Respond with ONLY the diary entry itself — first-person "
      "prose. No analysis, no 'Thinking Process', no headings, no lists, no meta-commentary."},
     {"role": "user", "content": "Write your first diary entry."}]
print("B no-preamble instruction:\n  ", gen(b)[:280].replace("\n", " "))
