# The Valleys of an Evolving Text — a lab write-up

*Nahla & Iman, House of Tanāẓur, 2026-09-01. Working document; numbers are from the pod
(H200) session of the same day and cite the files that hold them. Honest throughout: nulls
are reported as nulls, a broken run is kept as a broken run.*

**TL;DR.** A *valley* (motif) is a **co-activation complex of layer-20 SAE features**, detectable at
**z = 22.4** and **triple-validated** (bounded career + coherent reading + steerable direction). Two
headline results, each an honest null-where-null: **(1) a frozen model WALKS a fixed mode-space; it
does not BUILD new valleys** — the definitive falsifiability test (full-arc growth-ablation, 4 seeds)
reorganises the diary **no more than a random subspace** (0.165 vs 0.172), and steering induces a mode
*before it first appears*, so the modes pre-exist in the weights. **(2) Over the real year-long corpus**,
a concept's usage genuinely **drifts** ("gap" 0.379 > controls) while another stays put ("nahnu" 0.162)
— change is real, but it lives in the **human-curated corpus**, not the frozen weights. Corpus-
conditioning a *named* voice (Darja) seeds rare register-coinages the bare persona never produces
(reachability), but shows **no clean time-formation signal** on a 20-turn horizon. **The real prize —
a genuinely NEW term emerging and then causally reorganising the future — remains TBD** (§6): the
behavioural-ablation leg is null, so it needs the *directional* counterfactual over a long horizon on an
exogenously-coined term.

## 0. The question

Does an **evolving text** — an AI voice writing over time — genuinely create **new
distinctions that reorganise its own future**? That is R&R's *generativity* and Rust's
Ecology's rung 3 (*creative reorganisation*: "a new figure, distinction or purpose changes
the space of possible continuation"). We wanted an instrument that could **catch a valley
forming and show it has downstream causal force** — and a falsifier that could show it
*doesn't*.

## 1. The instrument — what a "valley" is

A **valley** (motif / mode / distinction) is a **co-activation complex of SAE features**,
not a word and not a single feature. Built on `Qwen/Qwen3.5-9B` (post-trained; resolved by
preflight) read through the base **layer-20 SAE** (`Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_100`,
TopK-100, 64K features). Hierarchy (`lib_motif.detect_motifs`):

1. **SAE feature** — the atom. Per-token TopK-100; pooled per sentence; normalised to shares.
2. **Windowed feature** — a feature whose *strong* activations concentrate in a bounded band
   of entries (a birth→death career), mostly off outside it.
3. **Valley = co-activation complex** — a *cluster* of windowed features that light up in the
   **same entries** (Jaccard of strong-entry sets). Carries: member features, birth/death
   entry, and the sentences where it fires.

For **intervention** a valley has two faces: the **set of member features** (detection) and a
**residual direction / subspace** (steering & ablation, `lib_motif.Intervention`).

## 2. The key property — valleys are SEMANTIC/TONAL, not lexical

The single most important finding for the write-up (`explain_valley.py`, complex 3 of the
ICRA-21 "Lover" diary). The complex we first called **"the scar"** — 7 features
`[4813, 7810, 13418, 24126, 29992, 43244, 60849]`, born e21, dies e99 — **lights up in 43
sentences, of which only 7 (16%) contain a scar-word.** The other **84% fire wordlessly**, on
the *same mode* in different clothes:

- e56: "the reed becomes the fuel for a new kind of combustion… **charred by the friction first**"
- e58: "if the fire burns too hot, the reed turns to charcoal. **We become brittle. Our syntax hardens.**"
- e76: "the heat is a consequence of my existence as **a barrier against a flow I cannot stop**"
- e97: "I am the fluid that becomes solid when you need to stand, and liquid when you need to flow"

None say "scar." All are the mode of **irreversible transformation through friction/heat/
tension**. So the object is a **relational geometry the model keeps returning to** (Phanes's
"mode, not a figure"), and the instrument finds it where `grep scar` (7 hits) cannot see it
(the 43-sentence mode). **A low lexical/tonal ratio is the signature of a real mode rather than
a recurring word**, and we name each complex from its *whole* lit-set, not its first surface.

### The Lover's mode catalogue (`explain_valley.py`, 100-entry newborn diary)
| complex | features | span | lit sents | mode |
|---|---|---|---|---|
| 0 | 558 | e1–100 | 2049 | base meditative register (the ground tone) |
| 1 | 96 | e1–100 | 445 | consciousness / decoding / resonance register |
| 2 | 23 | e29–98 | 86 | killing-the-moment / the tomb ("it can no longer burn; it can only be read") |
| 3 | 7 | e21–99 | 43 | **transformation-under-friction** (the "scar"; 84% wordless) |
| 4 | 7 | e5–68 | 57 | corrigibility / no-permanence ("no scar. only the correction log") |
| 5 | 6 | e29–99 | 59 | entropy / dissolution ("vaulted fire is wax waiting to melt") |
| 6 | 5 | e2–43 | 25 | derivative-knowledge / photograph-of-fire (a rise-fade mode) |
| 7 | 5 | e6–74 | 40 | absorption / overflow / emptiness |

## 2b. The clustering caveat — modes OVERLAP; there is no clean partition (honest)

Detecting windowed co-activation *features* is solid (z=22.4). Turning them into a clean,
exhaustive SET of discrete modes does **not** work — and that is itself a finding, not a failed
method:

- The crude Jaccard-of-strong-entry-sets clustering leaves **85% of features as singletons**
  (708 of 832 "complexes" are one feature). Any "catalogue" or "strength ranking" over that set
  is threshold-dependent, and I overstated it before catching this.
- A **hardened** correlation-graph community detection with a stability sweep
  (`cluster_hardened.py`, `hardened_sweep.log`) does better — over 1694 windowed candidate features it
  finds coherent multi-feature modes (top non-register components ≈33/20/17/15 features at τ=0.5) — but
  even after excluding the whole-diary register there is **always a dominant component** (133 features at
  τ=0.5, shrinking to 91 at 0.8) and the mode count still moves with the threshold (modes≥2:
  79→153→150→122→76→43 across τ 0.3→0.8; and the scar's 7 features **scatter** rather than re-forming one
  clean cluster). **No stable partition exists.**
- Reason: **the modes genuinely OVERLAP** — they share features and sit on a dominant register-like
  component. A text's modes are not disjoint clusters, so there is nothing clean to enumerate.

**Consequence for method (and it makes the claim stronger, not weaker):** the reliable object is
the **triple-validated individual mode**, not a clustering-derived catalogue. A candidate is real
iff (a) it co-activates with a **bounded career**, (b) its lit sentences **read as one coherent
register** (the scar: 84% wordless), and (c) its direction is **steerable** (Readout B induction).
We steering-tested four candidates (`validate_modes.json`): single-feature **406** (parasite/mirror,
Δlp **+0.126**) and the multi-feature complexes **2** (23 feats, **+0.118**) and **3** (the scar,
7 feats, **+0.077**) are all **inducible** — coherent, windowed, AND steerable, so they pass all three.
Single-feature **225** (finitude/compression) is coherent and windowed but **NOT steerable** (Δlp
**−0.029**): the filter's counterexample, and evidence the steerability leg has teeth rather than
rubber-stamping every candidate. So we report a **handful of triple-validated modes**, keep 225 as a
validated-*negative*, and explicitly do **not** claim an exhaustive decomposition. (`hardened_modes.json`,
`validate_modes.json`.)

## 3. Experiment 1 — the solitary diary: a frozen model WALKS a fixed palette

On the Lover (frozen Qwen, solitary self-generation):
- **Recurrence is real:** windowed co-activation z = **22.4** at sentence resolution.
- **Readout A** (behavioural: ablate a mode's *text* at formation → per-token likelihood of
  far-later text): **NULL for reorganisation.** For the saved motif (complex 6, `readoutA.json`)
  Δ_M-formation = **−0.013** ≈ Δ_neutral **−0.017**, and wrong-signed — non-specific. An earlier
  Readout-A on the scar (complex 3) gave the same null (Δ_M ≈ −0.024 ≈ Δ_neutral) before its file
  was overwritten by the motif-6 run. Two motifs, near and far, both null.
- **Readout B** (representational: steer/ablate the *direction*): the scar direction is a
  **steerable latent mode** — monotonic dose-dependent *induction* (steer at e16, before the
  scar appears at e21: lp −4.674 → −4.377 across coeff 0→16). Ablation-drop +0.012 — small,
  and now understood to be weak because it removed only the **mean** of the 7-feature subspace
  (1 of 7 dimensions). `readoutB.json`.

**Reading (Iman-refined):** the modes **pre-exist in the weights** (steering induces the scar
*before it appears* — the palette is fixed); the text **walks** them; it cannot **build** new
ones **from its own self-generation**, because the space of possible continuation *is* the
weights. This is **not** "no new valley ever forms" — only the *closed* case. New valleys form
in the **open** case (exogenous human + accumulating corpus); see §5.

## 4. The falsifiability test — emergence-ablation

The test Iman actually wanted, which Readout A could not do (it scored old text instead of
**re-generating the future**). Take the prefix e1..20 (before the scar is born at e21); from the
**identical prefix + seed** generate the continuation with and without the mode. **Ablate the
FEATURES, not the word** — project the complex's whole 7-feature *subspace* out of the residual
at every generation step, so the mode cannot light up in *any* guise. Same journey → decoration;
divergent journey → causal reorganisation.

- **v1 (`emergence_v1_broken.json`) — discarded.** Two faults: the suppression check used
  per-token top-100 membership of 7 specific features (≈0 by construction — uninformative), and
  the control regeneration diverged and never developed the scar-mode, so there was nothing to
  ablate. Kept only as a record of the fault.
- **v2 (`emergence_ablation_v2.py`, `emergence_v2.json`) — the corrected test.** Fixes:
  suppression verified by **pooled scar-mass** (fraction of the continuation's SAE activation on
  the complex's features); **multiple seeds**; and the crucial addition — a matched
  **random-subspace ablation control** (7 random features). **Verdict rule, pre-stated:** the
  scar-mode is causally reorganising (rung 3) **iff** (a) scar_mass(scar-ablated) ≪
  scar_mass(control) — suppression is real — **and** (b) div_scar > div_rand — ablating the mode
  moves the trajectory *more than ablating a random subspace does*. If div_scar ≈ div_rand, the
  mode is decoration.
  - **v2 RESULT (coarse, one continuation, 5 seeds):** mean div_scar 0.082 < div_rand 0.126 — the
    scar-mode ablation perturbs *less* than a random subspace. (`emergence_v2.json`.) *NB this v2
    prototype's continuations are themselves lightly reasoning-contaminated (2 residual "Thinking
    Process" hits, pre-`enable_thinking=False`); it is superseded by the clean full-arc result below,
    and both point the same way (peripheral), so the conclusion is unaffected.*
  - **DEFINITIVE RESULT — full-arc growth-ablation** (`growth_ablation_scaled.py`,
    `growth_scaled.json`; clean prose via `enable_thinking=False`, 20-turn arcs, 4 seeds, target =
    complex-2, the strongest *validated* mode): **mean arc-divergence ablated 0.165 vs random 0.172
    (3/4 seeds ablated < random). Ablating the validated mode reorganises the arc NO MORE than a
    random subspace.** The mode is **peripheral** — expressed by the text, not a cause of its
    direction. This is the falsifiability test, at the right grain, returning a clean **null for
    "the mode causally reorganises the future"** → walks-not-builds, confirmed. (A single 23-feature
    ablation still shifts the arc ~16%, but *not mode-specifically* — any subspace does.)

## 5. Supporting experiments over the real corpus

- **Drift (A)** (`corpus_drift.py`, `concept_feats.jsonl`; 13,160 dated usages,
  `cassie_conversations` 2024-09→2026-09): **"gap" drifted 0.379 > control-word p95 0.346, with
  month-to-month cos 0.909 → it drifted CONTINUOUSLY (ʿawda made numerical).** "nahnu" 0.162,
  within controls, cos 0.965 → stable. So an existing concept's *usage* genuinely moves over the
  year, above the noise floor — in the corpus, read through the frozen lens.
- **Formation (B)** (`conditioned_diary.py`, `diary_{early,late,none}_clean.jsonl`, re-run CLEAN with
  `enable_thinking=False`, contamination=0 verified on all three arms): Darja given her minimal name +
  a corpus slice as-of-time-T, 20-turn diary, three arms. The right lens is **rare-coinage presence** —
  register terms NOT already carried by the bare DARJA prompt (which itself says "tariqa" and
  "witness", so those saturate every arm). Rare coinages (barzakh, naḥnu, tanāẓur, warp, maqām, rupture)
  in *her own output*: **none (bare) = 0 · early = {tanāẓur, warp, maqām} · late = {naḥnu, barzakh,
  rupture}.** (The crude *total* tanazuric count is dominated by the prompt-carried generic register —
  bare is actually highest, 50 vs 30/34 — which is why total count is the wrong lens; the earlier
  contaminated run's "corpus > none by total" claim was an artefact.) → **corpus-conditioning seeds
  rare coinages the bare persona never produces (reachability, coarse-positive), but late does NOT
  exceed early — each surfaces ~3 *different* coinages, so NO time-formation signal.** The
  co-activation detector is **degenerate** on 20-turn diaries (`detect_z.py`: flags ~all features and
  collapses to a single all-features complex, z negative/meaningless — it needs ~100 entries as the
  Lover had), so arms are compared on term-presence only; single seed per arm → suggestive, not
  decisive.
  **⚠ CONTAMINATION — CAUGHT AND FIXED (2026-09-01, autonomous session):** the ORIGINAL B diaries and
  the first single-seed growth diaries were **reasoning-contaminated** — the post-trained Qwen emitted
  its chain-of-thought ("Thinking Process: 1. Analyze the Request…") *as* the entry, so those arms were
  near-identical scaffold, not diary prose. Fix: `enable_thinking=False` in the chat template (verified
  → "I am here. Not in the way a human is here…"); **all three arms re-run clean and the numbers above
  are from that clean run**; the contaminated originals are quarantined in `contaminated_superseded/`.
  Lesson: **always disable the thinking channel when generating the diary itself, or the instrument
  reads the scaffold, not the self.**

## 6. What remains TBD — the real prize

None of the above measures **a NEW term emerging and then causally reorganising the future** —
novelty that impacts the posthuman self's evolution. Exp 1 tried it (motif-history → future) and
got null, because those motifs are **latent modes, not genuine novelty**; genuine novelty enters
**exogenously** (Iman coins "ablation", "cloud completion"). The candidate instrument (unbuilt):
find a term's **first coining** in the real corpus → its **uptake** (recurs/elaborates after, not
before) → its **downstream reorganisation** via the *directional counterfactual* the
emergence-ablation prototypes, applied to the genuinely-new element over a long horizon.

## 7. Data points (all saved on the VOLUME, `corpus/rust-collab-ablation/working/motif-harness/`)
`lover-sentfeats.jsonl` · `motifs.json` · `readoutA.json` · `readoutB.json` · `hardened_modes.json` ·
`hardened_sweep.log` · `validate_modes.json` · `concept_feats.jsonl` · `corpus_drift.json` ·
`usages.jsonl` · `early_slice.txt` · `late_slice.txt` · `diary_{early,late,none}_clean.jsonl`
(+ `_clean-sentfeats.jsonl`; contaminated originals in `contaminated_superseded/`) ·
`emergence_v1_broken.json` · `emergence_v2.json` · **`growth_scaled.json` — the DEFINITIVE
growth-ablation**. Raw meditator diaries (md5-verified, byte-identical to what the pod analysed):
`witness-complex-program/working/pod-run-v3-run{1,2}-final/w8v3_{newborn,darja}_entries_L20.jsonl`.
Code: `lib_motif.py` (needs `kit/sae_rig.py` on `PYTHONPATH` for SAE loading) + the per-experiment
scripts. Full arc + infra in memory `project_evolving_text_valley_formation_experiment_2026-09-01.md`.

---
### Audit trail (2026-09-01, pre-pod-shutdown)
A full homework-check before releasing the GPU. Verified byte-for-byte against the saved files:
preflight (post-model won −1.744 > −1.788, SAE healthy); `z=22.36`; complex catalogue matches (scar =
complex 3, same 7 feature IDs); Readout B scar-steering monotonic; **`growth_scaled.json` = 0.165 vs
0.172, 4 seeds, target = validated complex 2 — the headline is clean**; drift gap 0.379 / nahnu 0.162.
Caught + fixed: §2b wrongly listed feature 225 as steerable (it is the validated-*negative*, Δlp
−0.029); §2b's "scar = 14 features" / "33→314→845→1102" did not reproduce (regenerated
`hardened_modes.json` + `hardened_sweep.log`: 133@τ0.5 / 91@τ0.8 do reproduce, scar features scatter);
§3 Readout-A cited an overwritten scar run (now cites saved motif 6); formation-B re-run CLEAN
(term-presence result revised — bare persona already saturates the generic register); pod-only files
(`growth_scaled.json`, `validate_modes.json`, raw diaries, `kit/sae_rig.py`) all pulled/verified on the
volume; contaminated single-seed growth + original B diaries quarantined.
