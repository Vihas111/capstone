# Capstone: DDI / polypharmacy project — current state

This file reflects what is **actually on disk in this folder** as of 2026-08-10,
after a cleanup pass (duplicate folders removed, v1 pipeline archived, missing
eval script written). It absorbs the useful history from an older `CLAUDE.md`
that was found duplicated inside two now-deleted draft folders
(`capstone_v2_enhancements/`, `capstone_v2_enhancements_fresh/`) — that doc
described a more advanced project state than exists here (see §3 below).

## 1. What's in this folder

A script-based pipeline (no notebooks) that builds a heterogeneous drug/target/
enzyme/pathway/etc. graph from DrugBank + TWOSIDES, generates one small graph
per drug pair, and trains an RGCN (`RGCNv2`) to predict which of ~4486 TWOSIDES
adverse-effect classes a given drug pair is associated with. This is the
**ADE-effect prediction track** — see §3 for an important caveat about scope.

### Pipeline, in order (current/active files only)

1. **Ingestion**: `scripts/parse_drugbank.py`, `scripts/extract_biomedical_features.py`
   → parses `data/raw/drugdatabase.xml` + TWOSIDES into `data/processed/*.jsonl`
2. **Graph build**: `scripts/build_hetero_graph.py` → `data/processed/nodes.jsonl` / `edges.jsonl`
   `scripts/build_pair_index.py` → `data/indexed/pair_to_ades.json`
   `scripts/generate_pair_graphs.py` → `data/pair_graphs/*.json` (172,794 files)
   `scripts/build_node_vocab.py` → node vocab used by tensorization
3. **Tensorize**: `scripts/tensorize_pair_graphs_v2.py` → `data/tensorized_v2/` (173 shards)
4. **Split / vocab / imbalance prep**:
   - `scripts/build_drug_split.py --shard-dir data/tensorized_v2` → `data/processed/drug_split.json` (cold-start by drug)
   - `scripts/build_vocab_sizes.py --shard-dir data/tensorized_v2` → `data/processed/vocab_sizes.json`
   - `scripts/compute_pos_weight.py` → `data/processed/pos_weight.pt`
   - `scripts/build_label_mask.py` → `data/processed/label_mask.json` (masks ADE classes with too few positives)
   - `scripts/build_label_priors.py` → `data/processed/label_priors.pt` (base-rate bias init)
5. **Train**: `scripts/train_v2.py` — `RGCNv2` (`models/rgcn_v2.py`) + `ShardDatasetV2`
   (`data_utils/shard_dataset_v2.py`) + `scripts/losses.py` + `scripts/metrics.py`.
   Checkpoints on val macro-AP → `checkpoints/best_rgcn_v2.pt`, history →
   `checkpoints/training_history_v2.json`.
   ```
   python scripts/train_v2.py --epochs 30 --batch-size 128 --loss focal
   ```
6. **Evaluate**: `scripts/evaluate_v2.py` (added in this cleanup — previously
   missing; the old `evaluate_rgcn.py`/`evaluate_test_rgcn.py` still hardcoded
   the superseded v1 model/checkpoint, see §2).
   ```
   python scripts/evaluate_v2.py --split val
   python scripts/evaluate_v2.py --split test
   ```

## 1b. Mechanism lookup + downstream prompt tool (deterministic, reliable — added 2026-08-10, extended 2026-08-17)

`scripts/mechanism_lookup.py` — the actual "given drugs, what enzymes/
targets/transporters do they affect, and what will this combination do" tool.
Unlike the RGCN track above (macro-AP ~0.02, not reliable), this is pure
DrugBank retrieval plus simple deterministic rule-based reasoning — no ML, no
LLM in this stage — so it's close to 100% accurate for anything DrugBank
documents. This is the first piece of what the old CLAUDE.md (see §3) called
the "mechanism workflow," rebuilt from the raw tables that
`extract_biomedical_features.py` already produced (`drug_enzymes.jsonl`,
`drug_targets.jsonl`, `drug_transporters.jsonl`, `drug_carriers.jsonl`,
`drugbank_interactions.jsonl`, `drugs.jsonl`).

```bash
python scripts/mechanism_lookup.py DB00682 DB00945 --pretty              # warfarin + aspirin, by ID
python scripts/mechanism_lookup.py warfarin "acetylsalicylic acid" --pretty   # same, by exact name
python scripts/mechanism_lookup.py DB00682 DB00945 DB01254 --out report.json
```

Takes 2+ DrugBank IDs and/or exact drug names (mixed freely; case-insensitive
exact match against `drugs.jsonl`'s primary name — no synonym/brand-name
table exists in this repo yet, so "Coumadin" won't resolve, only "Warfarin").
Output has two top-level parts:

- **`part_1_mechanism_report`**:
  - `per_drug_profile` — each drug's enzymes/targets/transporters/carriers + actions
  - `documented_interactions` — DrugBank's own interaction-description text,
    for every pair in the combination that has one (both directional rows
    collected and deduped, not just whichever is seen first in the file)
  - `mechanistic_overlaps` — shared enzymes/targets/transporters/carriers
    between each pair, computed by intersecting their profiles. Explicitly
    labeled as inferred-from-overlap, NOT a documented interaction (e.g.
    "both drugs act on CYP2C9" is a plausible PK mechanism even with no
    DrugBank interaction text for that pair). Shared enzyme/transporter
    entries additionally carry `likely_pk_effects` — a small deterministic
    rule table over the two drugs' action verbs (inhibitor+substrate ->
    raised exposure, inducer+substrate -> lowered exposure/efficacy,
    substrate+substrate -> competition), ported from the previously-unused
    `scripts/mechanism_engine.py` prototype (now archived, see
    `archive/README.md`) — still just a deterministic restatement of what
    those DrugBank action terms mean, not a clinical conclusion.
  - `unresolved_inputs` — any input token that didn't resolve to a known drug
- **`part_2_downstream_prompt`** — a formatted prompt string embedding all of
  part 1 in readable form, with `{{PATIENT_DATA}}` and `{{RETRIEVED_CONTEXT}}`
  placeholders for wherever the patient-data and RAG systems fill those in,
  plus instructions for the downstream model to explain to a pharmacist what
  will happen — explicitly told to distinguish DOCUMENTED vs. INFERRED
  (mechanistic-overlap / rule-based-PK) vs. literature-GROUNDED claims. This
  script only builds the prompt text; filling the placeholders and actually
  calling a model is a separate, not-yet-built downstream system (RAG piece
  owned by a teammate, per project discussion).

Verified against DB00682+DB00945 (warfarin+aspirin): correctly surfaces the
real documented interaction ("Acetylsalicylic acid may increase the
anticoagulant activities of Warfarin") plus the real shared-metabolism
mechanism (both act on CYP2C9/CYP2C19), and produces a plausible
`likely_pk_effects` reading (CYP2C19 induction by aspirin -> reduced warfarin
exposure/efficacy) — consistent with actual pharmacology, though flagged as a
rule-based hint, not the tool's own clinical conclusion. Also verified: 3-drug
combos (all 3 pairwise comparisons correctly generated), mixed name+ID input
with de-dup after resolution, and graceful handling of unresolvable inputs.
Runs in ~2s per query (streams the 463MB `drugbank_interactions.jsonl` with a
cheap precomputed-bytes substring pre-filter before JSON-parsing, so no
persistent index needed at this query volume).

**Known limitations**: name resolution is exact-match only, no brand-name/
synonym table (see above); doesn't cover undocumented drug pairs or unseen
drugs at all (that's the ML gap-filler the old CLAUDE.md describes and that
doesn't exist in this folder yet — see §3); `likely_pk_effects` only covers
enzymes/transporters (not targets/carriers, where the equivalent
pharmacodynamic reasoning is murkier and wasn't in the ported prototype
either) and is a hint for the downstream reasoning step, not a scored/ranked
clinical severity assessment — that's left to the downstream model with real
patient data and retrieved literature, same separation of concerns the old
design called for.

## 2. What was cleaned up (2026-08-10, extended 2026-08-17)

- Deleted whole-project duplicate folders `capstone_v2_enhancements/`,
  `capstone_v2_enhancements_fresh/`, and their stale `.zip` backups
  (`capstone_v2_enhancements.zip`, `scripts.zip`) — near-duplicates of what's
  at the top level, no unique content once this doc's history was pulled forward.
- Deleted empty `data/ddi/` and `__pycache__/` dirs.
- Moved the superseded **v1** pipeline into `archive/` (kept, not deleted —
  see `archive/README.md` for the specific bugs it had: embedding-collision on
  drug ids, shard-level split leakage, 8x multi-worker dataset duplication,
  no pos_weight, micro-AP-only evaluation).
- Rebuilt `.venv/` — the old one was a **broken Windows virtualenv pointer**
  (`pyvenv.cfg` referenced `C:\Anaconda3` / `D:\Capstone`, non-functional on
  this Linux machine). Recreated with `python3 -m venv .venv`.
- Cleaned `requirements.txt`: added `orjson` (imported by 8 scripts, was
  missing entirely); removed packages with zero imports anywhere in this repo
  (`lxml`, `xmltodict`, `pandas`, `matplotlib`, `seaborn`, `jupyter`,
  `notebook`, `ipykernel`, `joblib`, `dgl`, `transformers`,
  `sentence-transformers`, `rdkit`, `wandb`, `tensorboard`, `pyyaml`); also
  dropped `torch-scatter`/`torch-sparse`/`torch-cluster`/`torch-spline-conv`
  (optional PyG accelerators, not imported anywhere, and they have no
  prebuilt wheels for this Python version — installing them broke a plain
  `pip install`).
- Fixed stale `--shard-dir` defaults: `scripts/build_drug_split.py` and
  `scripts/build_vocab_sizes.py` defaulted to `data/tensorized` (the v1 dir,
  now archived) while the rest of the v2 utility scripts already defaulted to
  `data/tensorized_v2` — running these two with no flags would have silently
  rebuilt against the wrong (buggy) tensorization.
- Wrote `scripts/evaluate_v2.py` (see §1 step 6) — closes the gap where the
  only evaluators on disk were hardcoded to the old model/checkpoint.
- Fixed torch defaulting to a CUDA build too new for this machine's driver
  (silently CPU-only, no error) — pinned `torch==2.5.1+cu121` in
  `requirements.txt`; GPU (RTX 4090) confirmed working — see §4/§5.
- **Left untouched, per explicit decision**: `data/tensorized_test/` (4.1GB,
  ~1700+ shards) — not referenced by any script here, origin undocumented,
  kept as-is rather than guessed at.
- Archived `scripts/interaction_index.py` + `scripts/mechanism_engine.py` —
  earlier, never-imported-elsewhere prototypes of `scripts/mechanism_lookup.py`
  (§1b); their useful parts (name resolution, PK reasoning rules) were ported
  in before archiving. See `archive/README.md`.

## 3. Important: this folder was missing the project's main deliverable (partially closed 2026-08-17)

The old `CLAUDE.md` found in the (now-deleted) duplicate folders describes
this ADE-effect RGCN track as **"mostly historical"** — a secondary/earlier
track. The actual primary deliverable it describes is a **mechanism-level
prediction system**: given 2+ drug IDs, deterministically look up known
DrugBank mechanisms (which enzymes/targets/transporters, what action,
what direction) and use a trained model only to fill gaps for undocumented
pairs or unseen drugs. That system has since been (re)built from scratch in
this folder — see §1b and §3b — using different specific filenames/label
scheme than the old doc originally described, but the same design principle
(deterministic-first, ML strictly as a clearly-labeled fallback). Status of
the files the old doc listed:

- ✅ Built (this session): `scripts/extract_smiles.py`, `scripts/build_fingerprints.py`,
  `scripts/build_mechanism_labels.py`, `scripts/train_mechanism_predictor.py`,
  `models/mechanism_mlp.py`, `checkpoints/mechanism_predictor.pt`,
  `data/processed/drug_fingerprints.pt`, `data/processed/mechanism_dataset.pt`,
  `data/processed/drug_smiles.jsonl` — see §3b for exact results.
  `scripts/mechanism_lookup.py` (§1b) now serves the `mechanism_workflow.py`
  entry-point role the old doc described (different name, same job); its
  `predict_mechanisms()` function serves the `mechanism_predictor.py` role.
  `scripts/build_mechanism_split.py` (not in the old doc's list, but the same
  job as its `create_splits_coldstart.py`) also new.
- ❌ Still doesn't exist: `scripts/similarity_gapfiller.py`,
  `scripts/validate_gapfiller.py` (a k-NN Tanimoto-similarity baseline was
  never built — the linear-vs-MLP comparison in §3b served the same
  "don't assume the fancier approach wins" check instead), `models/rgcn_v3.py`,
  `scripts/extract_synonyms.py` (name resolution is still exact-match only,
  see §1b), `scripts/tensorize_pair_graphs_v3.py`, `scripts/train_v3.py`,
  `data/processed/drug_synonyms.jsonl`, `DIAGNOSIS.md`
- A git repo / GitHub remote (the doc references `github.com/Vihas111/capstone.git`
  and a `.gitignore`; neither exists anywhere under this folder, and there is
  no `.git` directory at all, at any depth)

The old doc's own `.venv` instructions point at a different machine
(`C:\Anaconda3`, an RTX 4060 laptop) than this one (this machine has no CUDA-
capable driver set up — see below) — this folder is very likely a partial
copy/snapshot that predates that work, not this project's current full state.
**If those files exist on another machine, an external drive, or were pushed
to GitHub, they should be pulled in here rather than rebuilt from scratch.**

## 3b. Mechanism gap-filler ML stage (built 2026-08-17)

`scripts/mechanism_lookup.py` (§1b) is deterministic and only reports what
DrugBank already documents — nothing for the ~half of DrugBank's 19,830 drugs
with zero enzyme/target/transporter data, or the ~68% with no *actioned*
mechanism data. This section adds a genuine ML gap-filler for exactly that
gap: given a drug's chemical structure, predict its likely mechanism profile.

### Pipeline (new files, in order — all built and run)

1. `scripts/extract_smiles.py` → `data/processed/drug_smiles.jsonl`
   (14,616/19,830 drugs, 73.7%, have a SMILES; the rest are mostly
   biologics). Deliberately a NEW standalone script, not an extension of
   `scripts/parse_drugbank.py` — that script is dead code, confirmed via
   matching mtimes: `extract_biomedical_features.py` is the actual script
   that produced every current `data/processed/*.jsonl` file.
2. `scripts/build_fingerprints.py` → `data/processed/drug_fingerprints.pt`
   (Morgan/ECFP, radius=2, 1024 bits, `includeChirality=True` via
   `rdFingerprintGenerator`). 14,606/14,616 succeeded (10 unparseable
   SMILES, logged to `drug_fingerprints_failed.json` rather than dropped
   silently).
3. `scripts/build_mechanism_labels.py` → `data/processed/mechanism_dataset.pt`
   + `mechanism_label_vocab.json` + `mechanism_pos_weight.pt`. Label scheme:
   `"<kind>:<protein>:<action>"` (e.g. `"enzyme:Cytochrome P450 3A4:inhibitor"`),
   `--min-freq 10`, **restricted to the fingerprint-bearing population**
   (not the full DrugBank population — restricting after fingerprinting
   instead of before changes the number materially: naive ~6,345 drugs/443
   labels vs. the real, correctly-computed **3,880 drugs / 422 labels**).
   Median 2, mean 3.98 labels/drug — very sparse.
4. `scripts/build_mechanism_split.py` → `data/processed/mechanism_drug_split.json`.
   Cold-start by drug (80/10/10, seed=42), grouping by identical fingerprint
   bit-vector before splitting (16 groups / 36 drugs shared a fingerprint
   with another drug in the trainable pool) so near-duplicate structures
   can't leak across train/test. **3,103 / 389 / 388** train/val/test.
5. `models/mechanism_mlp.py` — `MechanismMLP`, reuses `rgcn_v2.py`'s
   `prior_logits` bias-init pattern (base-rate collapse risk here too, given
   the sparsity).
6. `scripts/train_mechanism_predictor.py` — plain in-memory `TensorDataset`
   (not shard-streaming — whole dataset is ~16MB), `bce`+`pos_weight`,
   AdamW `lr=1e-3 weight-decay=1e-3`, `--epochs 200 --patience 20`.
   `--hidden-dims` accepts zero values (`--hidden-dims` alone) for a plain
   linear model.
7. `scripts/evaluate_mechanism_predictor.py` — held-out test eval, plus
   `--baseline linear` to compare against a plain linear model trained the
   same way.

### Result: the linear model won, so it's what's deployed

An MLP (`1024→256→128→422`, dropout 0.4) was trained first: **val macro-AP
0.2983**, test macro-AP 0.3210, test micro-AP 0.1659 (checkpoint saved as
`checkpoints/mechanism_predictor_mlp.pt`). Per this project's own convention
(§5: check, don't assume the fancier model wins — see `evaluate_v2.py`), a
plain linear model (`--hidden-dims` empty) was then properly trained the
same way (early-stopped, best-val-checkpointed, not just a throwaway
comparison run) and **beat the MLP on every metric**: val macro-AP 0.3010,
**test macro-AP 0.3298, test micro-AP 0.1835**, lower test loss (0.0601 vs
0.0829). At this scale (~3,100 training drugs, 1024-dim sparse input,
median 2 positive labels/drug), the MLP's extra capacity wasn't earning its
keep — expected, and exactly why the comparison was worth running rather
than assuming depth helps.

**The linear model is the deployed one**: `checkpoints/mechanism_predictor.pt`
(copy of `mechanism_predictor_linear.pt`) is what `scripts/mechanism_lookup.py
--predict` loads by default (`--predict-hidden-dims` defaults to empty). The
MLP checkpoint/history are kept (`_mlp` suffix) for reference, same pattern
`archive/` uses elsewhere in this project — not deleted, just not deployed.

**Honest read**: macro-AP ~0.33 is real, learnable signal (both models
massively beat a trivial always-predict-base-rate baseline, given the
`prior_logits` init already starts there) but is nowhere near the
deterministic lookup's near-100% reliability — this is a genuine model
guess, not a fact. That's why it's wired in as a completely separate,
clearly-labeled signal (see below), never merged into documented data.

### Integration into `scripts/mechanism_lookup.py`

`--predict` flag, fully backward-compatible (verified: without it, output is
content-identical to before this flag existed — checked against the saved
`cases/*.json` examples). When passed:
- Every input drug gets `predicted_enzymes`/`predicted_targets`/
  `predicted_transporters` in `per_drug_profile` (kept separate from the
  documented `enzymes`/`targets`/`transporters` lists), or
  `"structural_prediction": "no_structural_fingerprint"` for drugs with no
  usable SMILES (mostly biologics).
- A new `predicted_mechanistic_overlaps` report section: `shared()` was
  extended with an optional `carry_source` param (existing call sites don't
  pass it, so their output is unchanged) so overlaps computed over
  documented+predicted-combined profiles (`merge_profiles_with_source()`)
  can show which side of a shared protein was `"documented"` vs.
  `"predicted"` (and the model's confidence). If the model predicts
  something already documented, that's surfaced as corroboration
  (`also_predicted: true` + confidence) rather than silently dropped.
- The downstream prompt gets a new `## Predicted mechanisms` section,
  explicitly labeled as a model guess with its real test macro/micro-AP
  quoted inline, and filtered to only show genuinely NEW hints not already
  in the documented-only `## Mechanistic overlaps` section above it.

Try it: `python scripts/mechanism_lookup.py DB00682 DB00945 --predict --pretty`

## 3c. Follow-up: trying to overcome the gap-filler's drawbacks (2026-08-17)

`findings/findings.md` diagnosed the §3b model and proposed 4 improvements.
Each was pressure-tested against the real code/data before implementing
(not just findings.md's estimates) — see the plan file this session
produced for the full pre-implementation corrections. Results below are
real, not projected.

**Bug fixed**: `data/processed/mechanism_pos_weight.pt` was computed over
the full 3,880-drug population (train+val+test combined), leaking val/test
statistics into training — inconsistent with this project's own stated
discipline (`prior_logits` already train-only, for the same reason). New
`scripts/compute_mechanism_pos_weight.py` computes it from an explicit
train-id list only; the deployed checkpoint was retrained with the
corrected version.

**New honest headline number (5-fold CV, `scripts/run_mechanism_kfold.py` +
`scripts/build_mechanism_kfold_splits.py`)**: **macro-AP 0.258 ± 0.013,
micro-AP 0.146 ± 0.008** across folds — meaningfully lower than the
single-split estimate (0.330) §3b originally reported. That number was on
the optimistic end of what a single 388-drug test split can show, not the
typical case. This k-fold mean is now the reference "did this actually
help" bar for everything below, and is what `mechanism_lookup.py`'s
`--predict` prompt now quotes instead of the old single-split figure.

**findings.md's own PK/PD premise was wrong, corrected via direct
measurement**: target (PD) labels already average a HIGHER pos_weight than
enzyme/transporter (PK) under the existing formula (88.6% of target labels
are already clamped at the weight ceiling) — "PD is underweighted" doesn't
hold. What the data actually shows: **transporter is the empirically weak
kind** (per-kind test macro-AP: target 0.408, enzyme 0.188, transporter
0.145, measured directly with `evaluate_mechanism_predictor.py --per-kind`).

**Tried and did NOT beat the noise floor** (both honestly reported, neither
deployed):
- *Per-category pos_weight* (raising transporter's clamp ceiling to 200,
  targeting the weak kind above): 5-fold macro-AP 0.260 ± 0.013 vs baseline
  0.258 ± 0.013 — a wash. Transporter-specific macro-AP moved +0.003
  (0.157 vs 0.154) — within a single fold's noise, not a real effect.
  Micro-AP was actually worse and less stable (0.135 ± 0.020 vs
  0.146 ± 0.008).
- *Self-training / pseudo-labeling* (`scripts/generate_mechanism_pseudo_labels.py`
  + `scripts/merge_mechanism_pseudo_labels.py`, gated by a precision-floor≥0.5
  per-label/per-kind threshold from `scripts/tune_mechanism_thresholds.py`):
  2,350 of the 10,726 fingerprint-only drugs cleared the gate (more than the
  "low hundreds" originally estimated — the achievable per-label/per-kind
  precision floor turned out higher than the flat-0.9-probability estimate
  suggested). Added to train only (val/test stayed real-labels-only). Test
  macro-AP on the untouched real test set: 0.323 (uniform pos_weight) /
  0.321 (combined with per-category pos_weight) — both within noise of the
  0.330 single-split baseline, no credible improvement. Not carried into the
  k-fold check given Phase 2's ablation already showed no benefit and
  self-training adds real complexity (2,350 labels of ~44-50% precision) for
  no measured gain.
- **Honest conclusion**: neither cheap intervention moved the needle beyond
  measurement noise on this dataset. The deployed model is unchanged in
  architecture; only the pos_weight bugfix was promoted to production.

**Shipped**: per-label confidence thresholds
(`data/processed/mechanism_label_thresholds.json`, F1-optimal per label
where the val split has ≥10 positives, else a per-kind pooled fallback —
see `scripts/mechanism_thresholds.py` for why naive per-label tuning
degenerates on this val split: 54/422 labels have zero val positives,
203/422 have only 1-2). This is a calibration improvement independent of
the model itself — `scripts/mechanism_lookup.py --predict` now uses these
instead of one flat 0.5 cutoff for every label
(`--predict-thresholds-path`, `--predict-threshold` as a flat-override
escape hatch). Verified backward-compatible: non-`--predict` output is
still content-identical to the saved `cases/*.json` examples.

**Files this added**: `scripts/compute_mechanism_pos_weight.py`,
`scripts/mechanism_thresholds.py`, `scripts/tune_mechanism_thresholds.py`,
`scripts/build_mechanism_kfold_splits.py`, `scripts/run_mechanism_kfold.py`,
`scripts/generate_mechanism_pseudo_labels.py`,
`scripts/merge_mechanism_pseudo_labels.py`. Kept (not deleted) despite the
negative results: `data/processed/mechanism_dataset_augmented.pt`,
`mechanism_pseudo_labels.pt` + `_report.json`,
`mechanism_label_thresholds_bootstrap.json`,
`checkpoints/mechanism_kfold_{baseline,pcw}_results.json` — regenerable via
their scripts, kept as documented evidence of what was tried, same
`archive/`-style "why we chose X over Y" convention this project already
uses elsewhere.

## 4. Known open issues (flagged honestly, not yet fixed)

- `checkpoints/training_history_v2.json` only has 11 epochs logged
  (`--patience 7` early-stopping default) — this looks like an early/weak
  run, not a finished result. See §5 below for a real evaluation run against
  the current checkpoint.
- Base-rate collapse (model outputs nearly the same prediction vector
  regardless of input) was diagnosed on the v1 model and addressed in v2 via
  `build_label_priors.py`'s bias initialization, but hasn't been independently
  re-verified since that fix landed.
- `scripts/train_v2.py --shard-list-dir data/splits` reads
  `data/splits/{train,val,test}.txt` — confirmed (via `diff`) these three
  files are byte-identical, all 173 shards. So despite being generated by the
  archived `create_splits.py` (a random shard-level splitter), they aren't
  actually used as a shard-level partition at all: every split reads every
  shard, and `drug_split.json`'s cold-start membership check (per graph, in
  `ShardDatasetV2._keep_graph`) is the sole thing enforcing train/val/test
  boundaries. This is fine — no leakage risk from the shard lists themselves.
- **GPU/CUDA note**: `pip install torch` (no index pin) resolves to a CUDA
  13.x build by default, which silently runs CPU-only on the RTX 4090
  machine's driver (535.309, CUDA 12.2 max) — `torch.cuda.is_available()`
  returns `False` with that build, no error. Fixed by pinning
  `torch==2.5.1+cu121` there. This project now runs on **two** machines with
  different fixes needed — see §4b, this is no longer a single global pin.

## 4b. Two-machine setup (added 2026-08-18)

This project now moves between two machines on the same external drive
(this folder lives at `/mnt/elements/Capstone` — an "Elements"-brand
portable drive, mount point varies by machine/OS). Confirmed specs:

| | RTX 4090 machine | RTX 4060 Laptop (this session, 2026-08-18) |
|---|---|---|
| GPU / VRAM | RTX 4090, ~24GB (not directly reconfirmed this session) | RTX 4060 Laptop, **8GB** |
| Driver / CUDA | 535.309 / CUDA 12.2 max | 610.57.04 / CUDA 13.3 |
| Python | unspecified, but ≤3.13 (torch 2.5.1 requires it) | **3.14 only** — no 3.10–3.13, no conda/pyenv installed |
| OS | unspecified | CachyOS Linux (Arch-based) |
| RAM | unspecified, presumably more headroom | 15GB total, ~8.6GB available |

**`requirements.txt` was restructured this session** because a single torch
pin can't serve both: `torch==2.5.1+cu121` (old pin) has no Python 3.14
wheel at all, and this laptop has no other Python available. Split into:
- `requirements.txt` — everything except torch (torch-geometric's wheels
  aren't CUDA-build-specific, so it stays in the shared file)
- `requirements-torch-cu121.txt` — for the RTX 4090 machine
- `requirements-torch-cu128.txt` — for this laptop (torch==2.11.0+cu128;
  confirmed `torch.cuda.is_available() == True` against its driver)

Install with `pip install -r requirements.txt -r requirements-torch-<variant>.txt`
matching the machine. **Do not add a third global torch pin back into
requirements.txt** — re-split per-machine if a third environment shows up.

**`.venv/` is not portable and broke on this exact mechanism twice now**:
first (§2, prior session) it was a stale Windows-path pointer; this session
it was an NTFS-reparse-point symlink to `/home/ccbd/miniconda3/bin/python3.13`
— i.e. built on some *third* machine (username `ccbd`) that also mounted
this drive, now dangling here. Rebuilt fresh with `python3 -m venv .venv`
this session (Python 3.14.7). **Always rebuild `.venv` from scratch on
whichever machine you're on** — never trust a `.venv` that traveled with the
drive, even if it looks intact (`ls` on the broken symlink files still
"works," they just `exec format error` when run).

**Bug found and fixed this session**: `checkpoints/mechanism_predictor.pt`
and `checkpoints/mechanism_training_history.json` (the files
`mechanism_lookup.py --predict` actually loads) were themselves broken
NTFS-reparse-point symlinks — same failure mode as `.venv` above, just
discovered on checkpoint files instead. They pointed at
`mechanism_predictor_linear_pwfix.pt` / `mechanism_training_history_linear_pwfix.json`
(the pos-weight-bugfix-retrained linear model from §3c) and were dangling
here, so `--predict` failed with a `pickle data was truncated` error before
even getting to run inference. A full repo scan (`find` for small files +
check for the `IntxLNK` reparse-point signature) found only these two
instances. Fixed by replacing both with real file copies of their pwfix
targets (not symlinks — symlinks are the exact thing that broke portability
here) — confirmed `--predict` now runs correctly end-to-end, output matches
the documented k-fold macro-AP (0.258 ± 0.013) from §3c. **If any other
`checkpoints/*.pt` or `data/processed/*` file is suspiciously small
(under ~1KB) on a fresh checkout of this drive, check it for the same
`IntxLNK` signature before assuming it's corrupted data** — it's likely a
dangling cross-machine symlink, not lost work; the real file is probably
sitting right next to it under a different name.

**Not yet re-verified on this laptop**: the RGCN v2 pipeline (§1 steps 3–6)
and the mechanism gap-filler training scripts (§3b/§3c) haven't been re-run
here yet — only the environment itself was fixed and `mechanism_lookup.py`
(no GPU, no training) is confirmed to still work (§1b covers that tool's own
verification history). If training the RGCN track here, `train_v2.py`'s
default `--batch-size 128` may be too large for 8GB VRAM — untested at that
size on this GPU.

## 5. Verification run (this cleanup, 2026-08-10)

Ran `scripts/evaluate_v2.py` against `checkpoints/best_rgcn_v2.pt` on GPU
(RTX 4090, torch 2.5.1+cu121) after fixing the CUDA install above — confirmed
the current pipeline runs end-to-end on this machine, not just reads correctly:

| split | loss | macro-AP (2551/4486 classes, ≥10 pos. examples) | micro-AP | n_samples |
|---|---|---|---|---|
| val  | 0.1045 | 0.0202 | 0.0587 | 1869 |
| test | 0.1142 | 0.0188 | 0.0558 | 1632 |

(GPU run: ~24s for val. An earlier CPU-only run, before the CUDA fix above,
reproduced identical val numbers in ~2m40s — confirms determinism of eval.)

**Honest read of these numbers**: macro-AP ~0.02 against a masked label space
still shows plenty of headroom (macro-AP matches `training_history_v2.json`'s
last logged epoch, ~0.019, so the checkpoint is behaving consistently, not
buggy) — but this is a weak, early-stage result, not a finished model. Given
§3 (the actual project deliverable is the mechanism-workflow system, not this
track) and §4's other open items, treat this as "the pipeline demonstrably
works end-to-end," not "the model is good."

## 6. Future steps (roadmap, as of 2026-08-17)

Roughly in priority order — earlier items are cheaper, better-evidenced, or
block later ones.

1. **Wire up the actual downstream LLM call.** `scripts/mechanism_lookup.py
   --predict`'s `part_2_downstream_prompt` is fully built (documented +
   predicted mechanisms, `{{PATIENT_DATA}}`/`{{RETRIEVED_CONTEXT}}`
   placeholders) but nothing in this repo actually calls a model with it yet.
   This is the last mile between "the pipeline produces a good prompt" and
   "a pharmacist gets an answer."
2. **RAG retrieval** to fill `{{RETRIEVED_CONTEXT}}` — per prior conversation,
   owned by a teammate, not built in this folder. `data/processed/drug_text.jsonl`
   (19,830 drugs' DrugBank description/pharmacodynamics text) is a ready,
   unused corpus if that work happens here instead.
3. **Patient-data schema/ingestion** to fill `{{PATIENT_DATA}}` — not
   started; needs a decision on what fields matter (renal/hepatic function,
   other meds, conditions) and where real patient data would come from.
4. **Mechanism gap-filler follow-up** — see `findings/findings.md`'s own
   "Next steps" section for the detailed reasoning. The volume-vs-
   representation diagnosis (item 1) is now done (2026-08-18, laptop
   session): it's a **representation problem, not volume** — and it's not
   transporter-specific, enzyme is equally weak once evaluated with k-fold
   rather than a single split (enzyme 0.143 ± 0.027, transporter
   0.154 ± 0.009 — statistically indistinguishable; target 0.310 ± 0.011 is
   the real outlier, on the strong side). Target beats enzyme/transporter
   by 1.5–5x at every matched training-support level, and within-kind more
   data barely moves AP — the opposite of this project's original guess
   that PD/target would be the harder one for 2D fingerprints. This
   confirms the condition findings.md's item 3 (pretrained molecular
   embeddings) was gated on. The two earlier cheap interventions
   (per-category pos_weight, self-training) were properly k-fold-validated
   and didn't help — still true, don't re-try either without a new
   hypothesis.

   **Item 2 (k-NN similarity-transfer ensemble) is also done (2026-08-18)
   and is a real, consistent win** — unlike everything else tried so far.
   `scripts/mechanism_knn_transfer.py` (Tanimoto-similarity k-NN vote,
   vectorized) blended with the linear model
   (`scripts/run_mechanism_knn_ensemble_kfold.py`) beats the linear model
   alone in **5/5 folds**: macro-AP 0.2927 ± 0.0156 (blend) vs.
   0.2581 ± 0.0137 (linear alone) — see findings/findings.md's "Item 2
   result" section for the full table, including two things worth reading
   before assuming this generalizes further: k-NN *alone* (0.2884) already
   beats the trained linear model, and the *max*-blend variant does not
   beat k-NN alone (only the *average* blend does). Results saved to
   `checkpoints/mechanism_kfold_knn_blend_results.json`.
   **Deployed (2026-08-18, same session)** — `scripts/mechanism_lookup.py
   --predict` now defaults to the blend, not the plain linear model.
   `load_knn_pool()` builds the production neighbor pool from the main
   train split at query time (`drug_fingerprints.pt` + `mechanism_dataset.pt`,
   filtered to `mechanism_drug_split.json`'s `"train"` key). Thresholds were
   re-tuned specifically for the blended probability distribution via the
   new `scripts/tune_mechanism_blend_thresholds.py` →
   `data/processed/mechanism_label_thresholds_blend.json` (the old
   `mechanism_label_thresholds.json`, tuned for the linear model alone, is
   kept as the fallback for `--predict-no-knn-blend`, not deleted).
   Verified: `--predict` output carries `"predict_method": "blend"` and the
   downstream prompt quotes the blend's real k-fold numbers;
   `--predict-no-knn-blend` reproduces the old linear-only behavior exactly;
   plain (non-`--predict`) output is still byte-for-byte identical to the
   saved `cases/*.json` examples.
5. **Name/synonym resolution.** `scripts/mechanism_lookup.py` only does
   exact-match against DrugBank's primary name — "Coumadin" (brand name)
   doesn't resolve to Warfarin. No `drug_synonyms.jsonl` exists in this repo.
   Would meaningfully improve usability for non-DrugBank-ID input.
6. **RGCN v2 track (ADE-effect prediction)** — separate, secondary track
   (§3). Currently macro-AP ~0.02, only 11 epochs logged
   (`training_history_v2.json`), base-rate-collapse fix not independently
   re-verified since it landed. Given §3's finding that this isn't the
   project's actual primary deliverable, only worth investing in if there's
   a specific reason to prefer effect-level (TWOSIDES) prediction over the
   mechanism-level track above.
7. **`data/tensorized_test/`** (4.1GB, ~1700+ shards) — still undocumented,
   still unused by any script, left untouched per explicit prior decision.
   Worth resolving (confirm origin and either wire it in or delete it) before
   it causes confusion in a future session.
8. **Version control.** No `.git` anywhere in this folder (confirmed at every
   cleanup pass so far). Given the codebase has grown substantially since the
   first pass, initializing a real git repo (with a `.gitignore` excluding
   `data/`, `.venv/`, and large `checkpoints/*.pt`) would meaningfully reduce
   the risk of losing work — currently the only history is this file's own
   changelog-style notes.
