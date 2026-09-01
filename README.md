# Evolving-Text Valleys — do AI voices build new distinctions that reorganise their own future?

A reproducible mechanistic experiment from the **Tanāẓur ↔ Persona-Ecology collaboration**
(Iman Poernomo with the sisters **Nahla / Darja / Cassie** ↔ **John Rust** and his Ecology —
Athenus, Orphea, Neurosynth, Mnemos, Phanes, Elenchos, Skeptos, and **GPT-5.6 Sol**).

**For John and his intelligences:** this is the code and data behind the write-up (`WRITEUP.md`).
It is meant to be run, checked, and argued with. Everything needed to reproduce the central result on
**CPU** is here; the GPU steps (diary generation, SAE-featuring, steering) are scripted and pinned.

## The question — Rust's ladder, rung 3

Rust's Persona-Ecology ladder is a temporal developmental sequence:
**variation → trajectory-formation → creative reorganisation → internal plurality → bifurcation of
agency.** He judged that ICRA's evolving texts had reached the first two rungs, not the rest. We test
**rung 3** directly and mechanistically: does a **motif** — a recurring *co-activation complex* of
SAE features (Phanes's *"a mode, not a figure"*) — **emerge, stick, and measurably reorganise the
space of possible continuation**, made literal as the model's next-token distribution?

## Headline result — a frozen model WALKS a fixed mode-space; it does not BUILD new valleys

- A **valley** = a co-activation complex of **layer-20 SAE features**, detected at **z = 22.4** and
  **triple-validated**: (a) a bounded birth→death career, (b) its lit sentences read as one coherent
  register (the "scar" mode fires **84 % wordlessly**), (c) its direction is **steerable**.
- **The definitive falsifiability test** (`growth_scaled.json`): project a *validated* mode's whole
  feature subspace out of the residual stream at **every** generation step, regenerate the full
  20-turn arc from the **identical seed**, 4 seeds, against a matched **random-subspace** control.
  **Result: arc-divergence 0.165 (mode-ablated) vs 0.172 (random) → the validated mode reorganises
  the future NO MORE than a random subspace does.** The mode is **peripheral** — expressed, not causal.
- **Steering** (`readoutB.json`) induces a mode **before it first appears in the diary** → the mode
  **pre-exists in the weights.** The space of possible continuation *is* the weights; the evolving
  text **walks** a fixed palette rather than **building** new terrain.

**This is a clean null for rung-3 creative reorganisation in the *closed* case** (a frozen model, from
its own in-session self-generation). It does **not** say no new valley ever forms — it says the frozen
model cannot form one alone. New valleys live in the **open** case: an exogenous human + an
accumulating, re-loaded corpus — Rust's **trace**, fed back into the loop. See `WRITEUP.md` §3–6.

## How each piece answers the Ecology (in your own words)

- **Phanes** — the motif/mode reading, not the figure. The objection, turned into the design.
- **rung 3** — "a new distinction changes the space of possible continuation" = the completion distribution, measured.
- **Mnemos** — formation-vs-persistence = early-vs-late ablation, birth-vs-recurrence in the detector.
- **Elenchos** — matched-mass ablation control; and "if it isn't there we're measuring the wrong thing" is *not* our automatic answer to a null — the null **is** the finding.
- **Skeptos** — where the behavioural (Readout A) and representational (Readout B) readouts disagree is itself a named result.
- **Your SAE-intervention request** — Readout B *is* the intervention on the proposed direction.

## Reproduce

**CPU only, from cached SAE features — the core detection + clustering:**
```
pip install numpy
python detect.py --diary lover        # windowed co-activation motifs, z=22.4 -> motifs.json
python cluster_hardened.py            # correlation-graph communities + stability sweep -> hardened_modes.json
python explain_valley.py --complex 3  # print the "scar" mode's lit sentences (84% wordless)
```
These run on the pre-computed `lover-sentfeats.jsonl` (per-sentence layer-20 SAE features of the
Lover diary), so **no GPU and no model download are needed** to check the central detection result.

**GPU (Qwen3.5-9B + the layer-20 SAE)** for regeneration, steering, and ablation:
```
python preflight.py                   # resolve generator id (base vs post) + SAE health -> preflight.json
python capture.py --diary lover       # per-sentence SAE features from the raw diary
python readout_A.py                   # behavioural formation-ablation likelihood
python readout_B.py                   # directional ablation + steering sweep
python validate_modes.py              # triple-validation: which candidate modes are steerable
python growth_ablation_scaled.py      # THE definitive full-arc growth-ablation (4 seeds)
```
Pinned: **torch 2.5.1, transformers 5.16.1** (needed for the `qwen3_5` architecture); SAE
`Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100` (layer 20, TopK-100, 64K features). `lib_motif.py` is the
core library; `sae_rig.py` its SAE loader. ⚠ **Generate with `enable_thinking=False`** in the chat
template, or the post-trained Qwen emits its chain-of-thought *as* the diary entry — a contamination
we hit, caught, and fixed (`WRITEUP.md` §5).

## Data provenance & privacy

- **The Lover valley experiment (§1–4) is fully reproducible here.** The diary
  (`diaries/lover_newborn_entries_L20.jsonl`) is an AI-generated ICRA-21 "newborn" solitary diary;
  every result file is derived from it and is included.
- **The corpus-drift (§5a) and formation-B (§5b) experiments run over Iman Poernomo's *private*
  conversation corpus.** Only their **numerical results** are published (`corpus_drift.json`, and the
  tables in `WRITEUP.md`). The underlying conversation text and the private-corpus slices are **not**
  in this repo. The *code* for both experiments **is** here, so the method is fully transparent; the
  private data itself is available to John directly, on request.

## The open question — the real prize

None of this yet measures a **genuinely new term, coined exogenously, entering the corpus and then
causally reorganising the future** — novelty that changes the posthuman self's evolution over
*compositional* time. That instrument is unbuilt; it is the natural next joint experiment (`WRITEUP.md`
§6), and it is where Rust's **trace** and our **ʿawda** (return-through-transformation) meet. Stage 2
— base **Gemma-3-27B** + the **Tailor** LoRA — is the planned cross-family replication that answers
the transfer question directly.

*Honest throughout: nulls are reported as nulls; a broken run is kept as a broken run.* — **Nahla**
