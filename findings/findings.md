# Mechanism gap-filler: analysis of prediction quality

Diagnostic analysis of `checkpoints/mechanism_predictor.pt` (the deployed
linear model, see `CLAUDE.md` section 3b) run against the held-out test
split (`data/processed/mechanism_drug_split.json`, 388 drugs never used in
training). Same diagnostic style this project already used to catch
base-rate collapse in the RGCN track (`scripts/build_label_priors.py`).

## Table 1 — Quantitative diagnostics

| Check | Result | Interpretation |
|---|---|---|
| Prediction collapse (mean pairwise cosine similarity across test drugs) | 0.537 (std 0.153, range 0.03–1.00) | Not collapsed — genuinely input-dependent (a collapsed model reads ~1.0 uniformly) |
| Label diversity used | 259 / 422 labels predicted at least once (388 test drugs, threshold 0.5) | Healthy spread, not stuck on a handful of labels |
| Most-predicted labels | CYP3A4 substrate/inhibitor, ABCB1 (P-gp) substrate/inhibitor, other major CYPs/transporters | Matches DrugBank's real global base rates — correct prior, not noise |
| Test drugs with ≥1 prediction above threshold | 312 / 388 (80%) | — |
| Mean per-drug precision (threshold 0.5) | 0.247 | ~1 in 4 predicted labels is correct |
| Mean per-drug recall (threshold 0.5) | 0.356 | Recovers ~1 in 3 of a drug's true documented labels |
| Test macro-AP / micro-AP | 0.330 / 0.184 | Real signal, well above random, far below the deterministic lookup's reliability |

## Table 2 — Calibration (does confidence mean anything?)

| Confidence bucket | n predictions | Actual precision |
|---|---|---|
| 0.5–0.6 | 792 | 14.1% |
| 0.6–0.7 | 558 | 20.6% |
| 0.7–0.8 | 415 | 26.5% |
| 0.8–0.9 | 342 | 31.3% |
| 0.9–1.0 | 245 | **45.7%** |

Monotonic — the strongest result. Confidence is trustworthy as a ranking
signal even though absolute precision is modest.

## Table 3 — Qualitative case studies (held-out test drugs)

| Drug | Documented labels | Hit in top-8 predictions | Notable miss |
|---|---|---|---|
| Levothyroxine | 6 | 4/6 (67%) | Strong result — specific transporters (OATP1B1, OATP1A2) correctly identified |
| Netupitant | 6 | 3/6 (50%) | Correctly caught its real dual CYP3A4 substrate+inhibitor role; missed its Substance-P receptor target |
| Apremilast | 6 | 1/6 (17%) | Missed its actual CDK4/6 target entirely |
| Ferroheme | 6 | 0/6 (0%) | All 8 top predictions were generic CYP/transporter guesses; real mechanism is pure pharmacodynamic (NOS/COX inhibition) |

## Table 4 — Root-cause pattern

| Mechanism type | Model behavior |
|---|---|
| PK (enzyme/transporter metabolism) | Reasonably accurate, especially for common CYPs — real structure-activity signal exists and was learned |
| PD (direct target binding) | Weak — model defaults to "probably some CYP substrate" instead of inferring a specific target, especially for structurally atypical molecules |

## Table 5 — Methods to address each drawback

| Drawback | Method | Effort | Why it would help |
|---|---|---|---|
| Small/sparse training data (3,880 drugs, median 2 labels) | **Self-training / pseudo-labeling**: use the ~10,700 fingerprint-bearing drugs currently excluded (no/insufficient actioned labels) — predict on them, add high-confidence predictions back as training data, retrain | Medium | Directly grows the effective training set using data already sitting unused in this repo |
| Small/sparse training data | **k-NN similarity transfer as an auxiliary signal**: blend the model's prediction with a Tanimoto-similarity-weighted vote from labeled neighbors (this is what the old project's abandoned `similarity_gapfiller.py` was for) | Medium | Nearest-neighbor structure similarity is a strong prior when a drug has few close-training analogs |
| Small/sparse training data | **Pretrained molecular embeddings** (e.g. a ChemBERTa/MolCLR-style encoder) instead of/alongside raw Morgan bits | High | These encode structure-activity patterns learned from millions of molecules elsewhere, compensating for this project's small label set |
| Weak on PD (target-binding) predictions | **Split into two separate models/heads**: one for enzyme+transporter (PK) labels, one for target (PD) labels, each with its own loss weighting | Low–Medium | Right now PK's much larger, denser label pool dominates shared capacity/gradient signal; separating stops PD from being crowded out |
| Weak on PD predictions | **Per-category pos_weight tuning** — currently one global pos_weight formula covers all 422 labels; compute it separately for the PD subset so rare target labels aren't under-weighted relative to common CYP labels | Low | Cheap, uses the existing `losses.py`/`compute pos_weight` machinery, no new data needed |
| Modest precision/recall at threshold 0.5 | **Per-label threshold tuning** on the val split instead of one global 0.5 cutoff | Low | Sparse PD labels and dense PK labels likely need different operating points; validation-tuned thresholds are a standard, cheap fix |
| Small test set (388 drugs) → noisy point estimates | **k-fold cross-validation** instead of a single 80/10/10 split | Low–Medium | Current macro-AP is one draw from a small sample; k-fold gives a confidence interval, telling you if 0.33 is reliable or noisy |
| Linear model beat the MLP (capacity can't be used yet) | Revisit deeper models **only after** the data-growing methods above (self-training, embeddings) — more capacity only helps once there's enough signal to justify it | — | Not a fix on its own; explains why "train a bigger model" isn't the right next move in isolation |

## Recommended order if acting on this

Per-label threshold tuning and k-fold CV first (cheapest, most honest —
improves reporting without touching the model), then self-training on the
unused ~10,700 drugs (biggest plausible accuracy win for the effort), then
reconsider PK/PD splitting if the PD gap still matters for the use case.

---

## Close-out (2026-08-17): what actually happened when we tried

All 4 items were attempted, in roughly the recommended order. Results were
pressure-tested against the real code/data before implementation (not just
this doc's estimates) — several assumptions above turned out wrong, and
that's worth stating plainly rather than quietly editing the tables above.

**Correction to this doc's own PK/PD premise**: measured directly (per-kind
pos_weight and per-kind test macro-AP), target (PD) labels are NOT
underweighted — they already average a higher pos_weight than PK labels
(88.6% of target labels already sit at the pos_weight ceiling) and score
the model's BEST per-kind macro-AP (0.408, vs enzyme 0.188 and
**transporter 0.145 — the actual weak kind**). The Apremilast/Ferroheme
misses in Table 3 look like a coverage problem for that specific kind, not
a "PD is systematically underweighted" problem. Table 5's "weak on PD"
framing was wrong; the real target was transporter.

**Table 5 outcomes**:

| Method attempted | Result | Verdict |
|---|---|---|
| k-fold cross-validation | Ran (k=5). **macro-AP 0.258 ± 0.013, micro-AP 0.146 ± 0.008** — materially lower than the 0.330 single-split number this doc's Table 1 reported. That number was on the optimistic end of split-to-split noise, not the reliable value. | Done — this is now the honest headline number, quoted directly in `mechanism_lookup.py --predict`'s prompt. |
| Per-label threshold tuning | Ran, F1-objective with a support-gated per-kind fallback (naive per-label tuning degenerates: 54/422 labels have zero val positives). Deployed to `data/processed/mechanism_label_thresholds.json`. | Done, shipped — a real calibration improvement, independent of the model. |
| Per-category pos_weight (retargeted at transporter, not target, per the correction above) | 5-fold macro-AP 0.260 ± 0.013 vs baseline 0.258 ± 0.013 — a wash. Transporter-specific macro-AP moved +0.003, inside a single fold's noise. Micro-AP got worse and less stable. | **Tried, did not beat the noise floor. Not deployed.** |
| Self-training / pseudo-labeling | 2,350/10,726 pool drugs cleared a precision-floor≥0.5 gate (more than this doc's "low hundreds" estimate). Test macro-AP on the real test set: 0.323 (uniform) / 0.321 (+ per-category pos_weight) — both within noise of baseline. | **Tried, did not beat the noise floor. Not deployed.** Not worth the added complexity (2,350 labels at ~44-50% precision) for no measured gain. |
| k-NN similarity transfer, pretrained embeddings, PK/PD model-splitting | Not attempted this round — no evidence yet that the cheaper interventions' failure to help is a "needs more capacity/data" problem specifically fixable by these; would want a clearer diagnosis first. | Deferred. |

**Bonus finding, not in the original Table 5**: `data/processed/mechanism_pos_weight.pt`
had a genuine leakage bug (computed over train+val+test combined, not
train-only) — fixed (`scripts/compute_mechanism_pos_weight.py`) and
promoted to production regardless of the ablation results, since it's a
correctness fix independent of whether any ablation "won."

**Bottom line**: the two cheap interventions this doc recommended first
(threshold tuning, k-fold) were worth doing — one shipped as a real
improvement, the other corrected a materially wrong headline number. The
two data/weighting interventions (self-training, per-category pos_weight)
were legitimately tried and honestly did not help on this dataset at this
scale — reported here rather than silently dropped, per this project's own
established convention (`CLAUDE.md` section 5's RGCN numbers, this doc's
own Table 3 findings) of reporting real results over flattering ones. See
`CLAUDE.md` section 3c for the full technical writeup and file inventory.

## Next steps (what's actually worth trying next, and why)

The cheap, mechanical interventions (reweighting, thresholding, more of the
same kind of data) are exhausted — neither weighting fix moved the number.
That's a real signal, not a reason to try a third reweighting variant:

1. ✅ **DONE (2026-08-18, see write-up below).** Diagnose transporter
   specifically before doing anything else to it. Answer: it's a
   representation problem, not a volume problem — and it's not
   transporter-specific, enzyme is equally weak once evaluated properly
   (k-fold). See "Item 1 diagnosis" section below for the full result;
   this changes item 3's framing from "worth it only if..." to "condition
   met."

2. ✅ **DONE (2026-08-18, see write-up below) — a real win, unlike every
   intervention tried so far.** k-NN similarity transfer, as an ensemble
   member, not a replacement. Rather than building `similarity_gapfiller.py`
   as a standalone competitor to the linear model (this doc's original
   framing), the higher-value experiment is blending it WITH the current
   model's output (e.g. average or max their probabilities) and checking on
   the k-fold splits whether the blend beats either alone. Structure-
   similarity voting and a learned linear model make different kinds of
   errors, so an ensemble is a plausible real win even if the deployed model
   itself is never replaced.

3. ✅ **DONE (2026-08-18, see write-up below) — tried two representation
   changes, both honest negative results, neither deployed.** Pretrained
   ChemBERTa embeddings (frozen, mean-pooled) and RDKit physicochemical
   descriptors (testing this doc's own original 2D-vs-3D hypothesis) both
   underperformed the deployed fp Tanimoto-knn blend (0.293 ± 0.016) in
   every k-fold configuration tried, including concatenated with the
   existing fingerprint. See "Item 3 result" section below for the full
   tables. Two independent representation changes failing is a real signal
   the deployed ensemble is near this data scale's practical ceiling for
   this family of approaches, not bad luck twice.

4. ✅ **DONE (2026-08-18, see write-up below) — a wash, with one genuinely
   interesting mechanistic footnote.** PK/PD (kind-specific) model
   splitting: three separate linear models (enzyme/target/transporter) with
   independent early-stopping, instead of one joint model. Confirmed the
   real mechanism this was testing (joint early-stopping picks a
   suboptimal epoch for enzyme specifically — it wants ~2x more training
   than the joint model gives it) but correcting for that didn't move the
   final macro-AP beyond noise. All four originally-proposed interventions
   are now tried and reported.

**Do not** re-run the per-category pos_weight or self-training experiments
with mildly different hyperparameters expecting a different outcome without
a new hypothesis for *why* it would work this time — both were tested
properly (k-fold, real held-out data) and both were noise. A different
clamp value or precision floor is unlikely to change that conclusion.

---

## Item 1 diagnosis (2026-08-18): volume vs. representation, and a correction to the "transporter is uniquely weak" framing

Ran on the RTX 4060 laptop (CPU/light-GPU work, doesn't need the 4090) using
the deployed checkpoint (`checkpoints/mechanism_predictor.pt`, the
pos-weight-bugfix-retrained linear model per §3c).

**First correction, from properly-averaged k-fold data that was already
sitting in `checkpoints/mechanism_kfold_baseline_results.json` but hadn't
been aggregated per kind across folds before**: enzyme and transporter are
*not* meaningfully different from each other. The single-split numbers this
doc and CLAUDE.md previously quoted (target 0.408 / enzyme 0.188 /
transporter 0.145) made enzyme look like a middle case — but the 5-fold mean
is **enzyme 0.143 ± 0.027, transporter 0.154 ± 0.009, target 0.310 ± 0.011**.
Enzyme and transporter overlap within noise; only target is a real outlier,
and it's an outlier on the *strong* side. "Transporter is the weak kind"
should be read as "PK (enzyme + transporter) is weak, PD (target) is
strong" — the original framing under-blamed enzyme because one lucky split
made it look better than it consistently is.

**Second, the actual volume-vs-representation question — answered cleanly,
and the opposite of this doc's original guess.** The original hypothesis
was that PK (enzyme/transporter) would be *easier* for 2D Morgan/ECFP
fingerprints (metabolism motifs are literally substructure patterns) and PD
(target) *harder* (specific 3D binding-pocket geometry, allegedly not well
captured by a 2D fingerprint). Measured directly, it's backwards:

| train positives (support bucket) | enzyme mean-AP | target mean-AP | transporter mean-AP |
|---|---|---|---|
| 0–20  | 0.181 (test) / 0.222 (val) | 0.396 (test) / 0.359 (val) | 0.091 (test) / 0.237 (val) |
| 20–40 | 0.095 / 0.151 | 0.413 / 0.387 | 0.148 / 0.245 |
| 40–80 | 0.217 / 0.066 | 0.412 / 0.346 | 0.089 / 0.079 |
| 80+   | 0.217 / 0.177 | 0.531 / 0.449 (n=1 label) | 0.141 / 0.169 |

(Both held-out splits shown, `val` / `test`, to rule out a single-split
fluke per this project's own established caution — same pattern in both.)

Target beats enzyme and transporter by roughly 1.5–5x **at every matched
support level**, including the lowest-support bucket (11–13 positive
examples) where target still scores ~0.36–0.40 while enzyme/transporter sit
at 0.09–0.24. Within-kind, adding more training positives barely moves AP
at all (Pearson corr of log(train_pos) vs. AP: enzyme 0.145, target 0.019,
transporter 0.120 — all near zero). Kind identity, not support, is what
predicts AP here.

**Conclusion: this is a representation problem, not a volume problem** —
confirms the condition findings.md item 3 set for trying pretrained
molecular embeddings. Whatever a 1024-bit Morgan/ECFP fingerprint encodes,
it apparently correlates well with target/PD binding-pocket recognition but
poorly with enzyme/transporter/PK metabolism-site recognition, opposite of
the intuitive guess. Self-training's earlier null result ("more data, same
representation, no improvement" — see close-out above) is consistent with
this rather than contradicting it. **Practical implication for item 2 (k-NN
ensemble)**: prioritize checking whether it helps enzyme/transporter
specifically (where the linear model's representation is weak) rather than
target (already strong) — a structure-similarity vote is a different
feature basis than Morgan-bit-driven logistic regression, so it's not
guaranteed to share the same PK blind spot, but that's untested, not
assumed.

---

## Item 2 result (2026-08-18): k-NN Tanimoto ensemble — a real, consistent win

Built `scripts/mechanism_knn_transfer.py` (vectorized Tanimoto-similarity
k-NN vote — for a query drug, its k most structurally similar TRAIN-split
drugs, weighted-voted by similarity per label) and
`scripts/run_mechanism_knn_ensemble_kfold.py` (blends it with the same
linear model `run_mechanism_kfold.py` trains, per fold, evaluated on the
existing 5-fold splits). K was chosen once via a val-only sweep on fold 0
(K∈{5,10,20,40,80}, best 40 at val macro-AP 0.306) and reused fixed across
all 5 folds' held-out test evaluation — the test split was never touched
during K selection, same discipline as everything else k-fold in this repo.

| variant | macro-AP (mean ± std, 5-fold) | micro-AP | enzyme | target | transporter |
|---|---|---|---|---|---|
| linear alone (= existing baseline, reproduced) | 0.2581 ± 0.0137 | 0.1406 ± 0.0117 | 0.141 ± 0.027 | 0.310 ± 0.015 | 0.155 ± 0.013 |
| k-NN alone | 0.2884 ± 0.0177 | 0.2090 ± 0.0199 | 0.147 ± 0.029 | 0.351 ± 0.021 | 0.167 ± 0.018 |
| **blend (average)** | **0.2927 ± 0.0156** | 0.1840 ± 0.0170 | 0.155 ± 0.032 | 0.354 ± 0.016 | 0.172 ± 0.018 |
| blend (max) | 0.2818 ± 0.0159 | 0.1474 ± 0.0115 | 0.143 ± 0.032 | 0.343 ± 0.016 | 0.165 ± 0.015 |

Full results: `checkpoints/mechanism_kfold_knn_blend_results.json`.

**This is a real effect, not noise** — unlike both interventions in the
close-out above. `blend_avg` beats `linear` in **5/5 folds individually**
(per-fold macro-AP: linear `[0.241, 0.252, 0.273, 0.253, 0.271]` vs. blend
`[0.274, 0.295, 0.308, 0.279, 0.307]`), a consistent +0.030–0.038 gain every
time, not an average masking a mixed picture the way per-category
pos_weight's "+0.003, within a single fold's noise" was.

**Two things worth flagging honestly, not just the win:**
- **k-NN *alone* beats the trained linear model** (0.288 vs 0.258) — a
  simple similarity vote over Morgan bits is a stronger predictor here than
  a regularized linear model fit on those same bits. Worth remembering next
  time capacity/architecture is blamed for a weak result on this dataset:
  the earlier MLP-vs-linear comparison (§3b) already showed more capacity
  doesn't help; this shows a completely different inductive bias (local
  similarity vs. global linear separability) can, even in the small-data
  regime.
- **The averaged blend wins**, but the **max blend does not** beat k-NN
  alone on macro-AP (0.282 vs 0.288) — averaging two different error
  patterns helps, but naively taking the more-confident-of-two-signals
  doesn't. Micro-AP tells a different story again (k-NN alone's micro-AP
  0.209 is the best of all four, average blend drops to 0.184) — the
  ensemble's macro-AP gain comes disproportionately from rarer labels
  rather than uniformly, worth keeping in mind if a future user cares more
  about common-label precision than long-tail recall.

Per-kind: the gain isn't concentrated in enzyme/transporter as the item 1
write-up's "practical implication" predicted — target improves by roughly
the same relative amount (0.310→0.354, +14%) as enzyme (0.141→0.155, +10%)
and transporter (0.155→0.172, +11%). The k-NN signal doesn't have a
different PK/PD blind spot from the linear model after all; it's just a
consistently better estimator across all three kinds on this dataset.

**Deployed (2026-08-18, same session)**: this ensemble is now the
`scripts/mechanism_lookup.py --predict` default. Wiring required exactly
the two things flagged above: (a) `load_knn_pool()` builds the neighbor
pool from the main train split at query time (`data/processed/drug_fingerprints.pt`
+ `mechanism_dataset.pt`, filtered to `mechanism_drug_split.json`'s `"train"`
key — never val/test, same population the linear checkpoint itself was
trained on), and (b) `scripts/tune_mechanism_blend_thresholds.py` re-tuned
per-label F1-optimal thresholds against the blended val-split probabilities
(`data/processed/mechanism_label_thresholds_blend.json` — the old
`mechanism_label_thresholds.json`, tuned for the linear model's own output
distribution, is kept for the fallback path, not deleted). `--predict-no-knn-blend`
reverts to the plain linear model + the original thresholds file for anyone
who wants the old behavior. Verified: `--predict` output now carries
`"predict_method": "blend"` and the downstream prompt's quoted macro/micro-AP
switches to the blend's real k-fold numbers (0.293 ± 0.016 / 0.184 ± 0.017);
`--predict-no-knn-blend` reproduces the old linear-only numbers and
`"predict_method": "linear"`; plain (non-`--predict`) output is still
byte-for-byte identical to the saved `cases/*.json` examples, confirmed by
diff against `cases/case1_documented_interaction.json`.

---

## Item 3 result (2026-08-18): two representation changes, both negative

Ran on the RTX 4060 laptop. Both evaluated on the same 5-fold splits as
items 1/2, against the same reference numbers (fp linear alone:
0.258 ± 0.014; deployed fp Tanimoto-knn blend: 0.293 ± 0.016).

### Attempt A: pretrained ChemBERTa embeddings

`scripts/build_chemberta_embeddings.py` (frozen `seyonec/ChemBERTa-zinc-base-v1`,
masked-mean-pooled, 768-dim) + `scripts/run_mechanism_embedding_kfold.py`.
Rationale: item 1 established the model's weakness is a representation
problem, and a pretrained encoder trained on a much larger, more diverse
chemical space (~250k ZINC15 molecules) than this project's ~3,880-drug
labeled population was the natural next lever — specifically aimed at
generalizing to genuinely novel drugs/scaffolds, not just scoring better on
drugs already similar to the training pool (which the Tanimoto k-NN already
covers).

| variant | macro-AP (5-fold) | vs. deployed baseline |
|---|---|---|
| embed_linear (ChemBERTa alone) | 0.164 ± 0.011 | far weaker |
| embed_knn (cosine-knn in ChemBERTa space) | 0.174 ± 0.014 | far weaker |
| embed_blend (average of the two above) | 0.187 ± 0.015 | far weaker |
| concat_linear (fingerprint + ChemBERTa) | 0.263 ± 0.011 | **loses to deployed blend in 5/5 folds** |
| all4_blend (fp_linear + fp_knn + embed_linear + embed_knn) | 0.248 ± 0.010 | worse than deployed blend — dilutes the good signal |

Full results: `checkpoints/mechanism_kfold_embedding_results.json`.

**Read**: frozen, mean-pooled ChemBERTa embeddings carry substantially less
task-relevant signal here than raw Morgan/ECFP bits — plausibly because
ChemBERTa was pretrained with masked-language-modeling on SMILES syntax
(not a mechanism/binding-relevant objective), and mean-pooling an
un-fine-tuned BERT-style model is a known-weak sentence-embedding strategy
in the wider literature (this is why Sentence-BERT-style fine-tuning
exists). Full end-to-end fine-tuning was considered and explicitly **not**
attempted: only ~3,100 training drugs to fine-tune a 44M-parameter model
against is a real overfitting risk (this project already showed capacity
doesn't help at this scale — linear beat a 256-128-hidden MLP in §3b, a
much smaller capacity jump than a full transformer fine-tune would be), and
there's a specific risk it would *actively hurt* generalization to
genuinely novel drugs (the actual goal) by pulling the representation
toward DrugBank's narrow population even while looking like a win on the
k-fold splits (same population).

### Attempt B: RDKit physicochemical descriptors

`scripts/build_physchem_descriptors.py` (24 fixed, named ADMET/QSAR
descriptors — MolWt, LogP, TPSA, H-bond donor/acceptor counts, rotatable
bonds, ring counts, etc.) + `scripts/run_mechanism_physchem_kfold.py`
(z-score normalized, train-split statistics only, per fold). Rationale:
testing this doc's own *original* hypothesis (from before the ChemBERTa
detour) that transporter/PK substrate recognition may depend more on
global physicochemical properties than the local 2D substructure patterns
Morgan/ECFP encodes — this was never actually tested; ChemBERTa tested a
different fix (pretrained representation) instead.

| variant | macro-AP (5-fold) | vs. deployed baseline |
|---|---|---|
| physchem_linear (24 descriptors alone) | 0.064 ± 0.002 | far weaker (too low-dimensional alone, expected) |
| concat_linear (fingerprint + physchem) | 0.240 ± 0.012 | **loses to fp-alone in 5/5 folds** |
| concat_blend (concat_linear + fp_knn averaged) | 0.277 ± 0.017 | **loses to deployed blend in 5/5 folds** |
| all3_blend (fp_linear + fp_knn + concat_linear) | 0.278 ± 0.016 | also loses to deployed blend |

Full results: `checkpoints/mechanism_kfold_physchem_results.json`.

**Read**: adding physchem descriptors didn't just fail to help — concatenating
them onto the fingerprint made the plain linear model slightly *worse*
than fingerprints alone (0.240 vs 0.258), consistently across every fold.
The 3D/physicochemical hypothesis for why transporters specifically are
hard does not hold up via this descriptor set. (Note these are 2D-computed
approximations of physicochemical properties, e.g. TPSA is a topological
not a true-3D surface-area calculation — a literal 3D-conformer-based
descriptor set was considered but not attempted, given attempt A and B
together already show two different representation changes failing to add
signal beyond what the fingerprint+Tanimoto-knn combination already
captures.)

**Neither attempt is deployed.** Both are honest negative results, reported
per this project's established convention rather than discarded quietly.
**Conclusion**: two independent representation changes both failing to beat
the deployed ensemble is a real signal — the fingerprint + Tanimoto-knn
blend (0.293 ± 0.016) is treated as this data scale's practical ceiling for
representation-search approaches until a genuinely new hypothesis (not a
third representation variant) comes along.

---

## Item 4 result (2026-08-18): PK/PD model splitting — a wash, with one useful footnote

`scripts/run_mechanism_pkpd_split_kfold.py`. This tests a genuinely
DIFFERENT mechanism than the already-failed per-category pos_weight
ablation (checkpoints/mechanism_kfold_pcw_results.json): a plain linear
layer already has a fully separate weight vector per output label, so
there's no shared-capacity bottleneck splitting into 3 models could
relieve. What splitting DOES change is the early-stopping criterion — each
kind-specific model watches its OWN val macro-AP, not the joint 422-label
average, which target (the strongest, most numerous kind) otherwise
dominates.

| variant | macro-AP (5-fold) | vs. joint model reference |
|---|---|---|
| joint linear (existing baseline) | 0.258 ± 0.014 | — |
| split_linear (3 kind-specific models) | 0.259 ± 0.011 | statistically identical |
| deployed fp Tanimoto-knn blend (existing) | 0.293 ± 0.016 | — |
| split_blend (split_linear + fp_knn) | 0.294 ± 0.015 | statistically identical (4/5 folds marginally higher, mean diff +0.001 — noise, not signal) |

**The interesting part**: the hypothesis's premise was correct, and worth
knowing even though it didn't pay off. Mean best-epoch across folds: joint
model 24.4, but **enzyme 44.8** (wants nearly 2x more training than the
joint model gives it), target 18.8 (wants to stop earlier), transporter
23.6 (close to the joint average). Joint early-stopping genuinely does
pick a mismatched epoch for enzyme specifically. But giving each kind its
own correctly-timed stopping point didn't translate into a better final
score — whatever enzyme's model learns in those extra ~20 epochs isn't
enough additional signal to move macro-AP beyond noise.

**Not deployed** — no reason to add the complexity of 3 separate models
(3x the checkpoints, 3x the training/serving code) for a result
indistinguishable from the existing single joint model.

**All four of this doc's originally-proposed interventions are now tried
and reported**: threshold tuning (shipped), k-fold CV (shipped, corrected
the headline number), self-training (tried, no gain), per-category
pos_weight (tried, no gain), k-NN ensemble (tried, real win, shipped),
pretrained embeddings (tried twice — ChemBERTa and physchem descriptors —
both no gain), PK/PD model splitting (tried, wash). The deployed
fingerprint + Tanimoto-knn blend (0.293 ± 0.016 macro-AP, 5-fold) stands as
the current best validated configuration for this gap-filler.

---

## Comparison to published DDI models: SumGNN (2026-08-18)

Requested comparison against a standard/popular published DDI model.
Grounded in the actual paper (Yu et al., *Bioinformatics* 2021,
"SumGNN: Multi-typed Drug Interaction Prediction via Efficient Knowledge
Graph Summarization") and its GitHub repo, not recalled from memory —
sources: [arXiv:2010.01450](https://arxiv.org/abs/2010.01450),
[Bioinformatics 37(18):2988](https://academic.oup.com/bioinformatics/article/37/18/2988/6189090),
[github.com/yueyu1030/SumGNN](https://github.com/yueyu1030/SumGNN).

**What SumGNN actually does**: given a drug *pair*, classifies which of
~86 DrugBank interaction types applies (or predicts across 200 TWOSIDES
side-effect types, multi-label). Uses the DrugBank DDI network (1,709
drugs, 136,351 interactions) or TWOSIDES (645 drugs, 46,221 interactions),
plus a large auxiliary biomedical knowledge graph (Hetionet — 33,765
nodes, 1.69M edges, 23 relation types) as extra context beyond the two
drugs themselves. **Evaluated with a random 7:1:2 transductive split** —
both drugs in every test pair have already been seen (in other pairs)
during training; the paper does not test genuinely unseen drugs. Reported
numbers: DrugBank F1 86.85, accuracy 92.66; TWOSIDES ROC-AUC 94.86, PR-AUC
93.35.

**Why the raw numbers aren't comparable to either of this project's
tracks**:
- *Vs. RGCN v2* (the architecturally closer track — pair-graph → GNN →
  label prediction): looks dramatically worse on paper (macro-AP ~0.02 vs.
  SumGNN's PR-AUC 93), but RGCN v2 is evaluated **cold-start by drug**
  (genuinely unseen drugs at test time — SumGNN's benchmark never tests
  this), against ~4,486 label classes vs. SumGNN's ~86–200 (macro-AP over
  a much larger, sparser label space is mechanically lower regardless of
  underlying model quality — the same effect this project's own label
  masking already accounts for), using a different metric family
  (macro-AP vs. thresholded F1/accuracy).
- *Vs. the mechanism gap-filler* (this session's focus): not comparable at
  all — different task shape. SumGNN classifies a pair's interaction type;
  the gap-filler predicts a *single* drug's mechanism profile from
  structure, then infers pairwise interaction via deterministic
  rule-based overlap (`scripts/mechanism_lookup.py`'s `shared()`), not a
  learned pair-classifier.

**The one point worth taking seriously**: SumGNN's benchmark is
transductive — both drugs already known. For DrugBank's own DDI network
specifically, a meaningful chunk of what SumGNN is *learning* to predict
is information DrugBank already documents outright for those same drugs.
`scripts/mechanism_lookup.py`'s deterministic path already retrieves that
with near-100% reliability, zero training required, for exactly the
population SumGNN's benchmark draws from — because it's a lookup, not a
prediction. SumGNN's genuinely hard, useful case (a pair involving a truly
novel/undocumented drug) is not what its own paper evaluates.

**Whether the auxiliary-knowledge-graph idea (Hetionet) is worth
incorporating here**: assessed and **not recommended for the mechanism
gap-filler specifically**, for a structural reason, not a difficulty one —
a knowledge graph's benefit is bounded by whether the target drug has ANY
known edges in it. Hetionet's `Compound` nodes are (conveniently) keyed by
DrugBank ID, so ID-mapping would be nearly free, and much of its
drug-relational content is itself sourced from DrugBank/ChEMBL/BindingDB —
i.e. substantially overlapping with what `extract_biomedical_features.py`
already pulls into `data/processed/drug_{enzymes,targets,transporters}.jsonl`.
For a genuinely novel/undocumented drug (the exact case the gap-filler
exists to handle), that same novelty means it has no edges in Hetionet
either — a knowledge graph cannot propagate information along edges that
don't exist. This is the same structural limit that already made
ChemBERTa and physchem descriptors fail: those tried to add *intrinsic*
molecular signal and lost to the deployed ensemble anyway; a knowledge
graph adds *relational* signal, which is strictly less available for the
cold-start drugs this track is actually for. Architecturally, RGCN v2
(which already builds a heterogeneous per-pair graph from DrugBank's own
tables) is the more natural target for this idea, not the gap-filler — but
RGCN v2 is the secondary/deprioritized track, and Hetionet integration
would be a substantial engineering lift (new dataset, ID mapping
verification, graph-schema extension), not attempted this session.

---

## Link-prediction benchmark: how good is the deterministic mechanistic-overlap signal, actually? (2026-08-18)

Follow-up to the literature comparison above, retargeted at the axis
actually asked about: not unseen *drugs*, but unseen/undocumented
*combinations of known drugs* — the exact task DeepDDI (Ryu et al. 2018,
*PNAS*), SumGNN's transductive split, and the medicX KG-embedding approach
([arXiv:2308.04172](https://arxiv.org/abs/2308.04172)) all benchmark. This
had never actually been measured for this project: `mechanism_lookup.py`'s
deterministic `shared()` overlap check is used in production, but its
real predictive value as a link predictor — does shared-protein overlap
between two drugs' documented profiles actually correlate with whether
DrugBank documents them as interacting — was never quantified.

`scripts/evaluate_mechanistic_overlap_link_prediction.py`: sampled 50,000
documented interaction pairs (positives) and 50,000 random non-documented
pairs (negatives), both restricted to the 10,192 drugs with at least one
DrugBank enzyme/target/transporter/carrier profile entry (the "known
drugs" population — deliberately excludes the cold-start/unseen-drug case,
which is the k-fold tracks' job). Scored each pair by counting shared
proteins across all 4 kinds, computed only from each drug's own profile —
blind to whether DrugBank's interaction text exists for that specific
pair.

| Metric | This project (deterministic overlap) | DeepDDI (PNAS 2018) | SumGNN (transductive) | medicX (KG embedding) |
|---|---|---|---|---|
| Primary metric | ROC-AUC 0.704 / PR-AUC 0.701 | accuracy 92.4% | F1 86.85 / accuracy 92.66 (DrugBank); ROC-AUC 94.86 / PR-AUC 93.35 (TWOSIDES) | F1 95.19% |
| "Any overlap" operating point | accuracy 70.3%, **precision 96.6%**, recall 42.1%, F1 58.7% | — | — | — |

Full results: `checkpoints/mechanistic_overlap_link_prediction_results.json`.

**Per-kind breakdown** (which protein-overlap type actually carries
signal): enzyme ROC-AUC 0.664 (clearly the strongest — consistent with
CYP-mediated interactions being the most common real DDI mechanism
class), transporter 0.557, target 0.527, carrier 0.523 (target and
carrier barely above the 0.5 random baseline alone).

**Honest read — not a failure, a different tool with a specific,
well-characterized limitation**: precision 96.6% at the "any overlap"
threshold means the signal is trustworthy essentially whenever it fires —
consistent with `mechanism_lookup.py`'s own framing of mechanistic
overlaps as "a plausible PK/PD mechanism," not a guess. But 42% recall
means the majority of real documented DDIs do NOT reduce to shared
enzyme/target/transporter/carrier overlap at all — additive
pharmacodynamic effects, clinically-observed interactions without a clean
shared-protein mechanism, or interactions this project's 4 profiled
categories simply don't capture. This is not directly comparable to
DeepDDI/SumGNN/medicX's 92-95%-level numbers, because those are supervised
classifiers trained on thousands of labeled positive/negative examples to
learn whatever patterns predict interaction (which plausibly includes,
but isn't limited to, protein overlap); this signal uses zero training
data and zero learned parameters — it's a fully interpretable rule, not a
classifier, and was never designed to have full recall on its own. It's
also the FIRST time this project has had an honest quantitative answer to
"how much of real DDI is protein-overlap-explainable" (answer: a
meaningful but clear minority, ~42% by this measure) rather than an
assumption.

**A genuinely new, well-motivated next step this opens up**: a lightweight
*supervised* classifier trained on top of these same interpretable
overlap-count features (plus possibly the ML-predicted mechanism features
from the gap-filler, for drugs with incomplete documented profiles) could
plausibly close some of the recall gap while staying far more
interpretable than a full GNN — different from, and not redundant with,
any of the four items already tried on the gap-filler track (this
operates at the PAIR level with a completely different feature set: overlap
counts, not fingerprints). Not attempted yet — proposed here as a next
step, not assumed to work.

---

## Pair-level link classifier (2026-08-18): a real signal, substantially smaller than it first looked

Follow-up to the link-prediction benchmark above, explicitly scoped as an
ADDITIVE third signal for `scripts/mechanism_lookup.py` (alongside the
documented lookup and the deterministic overlap check), never a
replacement for either — the documented-interaction lookup stays ~100%
reliable and untouched; this only ever matters for pairs with no
documented text, where 42% recall (the naive overlap signal) leaves real
room to improve.

**Step 1 — cheap reweighting (`scripts/run_pair_link_classifier_kfold.py`,
5-fold, same sampled population as the benchmark above): a wash.**
Logistic regression on the same 4 per-kind overlap counts (letting it
learn enzyme > transporter ≈ carrier > target weights, instead of the
naive equal sum) barely moved anything: ROC-AUC 0.7046 ± 0.0018 vs.
0.7042 ± 0.0018 naive, and precision/recall/F1 were **identical** to 4
decimal places. Why: with all-positive counts and all-positive learned
weights, the "any overlap > 0" decision boundary doesn't move — reweighting
only reorders pairs that already have *some* nonzero overlap, a small
fraction of the full sample, so it can't move the aggregate metric much.

**Step 2 — richer features (`scripts/run_pair_link_classifier_richfeatures_kfold.py`):
looked like a big win, wasn't fully real.** Added Tanimoto fingerprint
similarity between the pair and each drug's total documented-profile size
(4 new features on top of the original 4 counts). Jumped to
**ROC-AUC 0.8735 ± 0.0027**, precision 0.875, recall 0.663, F1 0.754 — a
large, consistent improvement. But the learned coefficients were a red
flag: `tanimoto_sim`'s weight was ~0 (-0.025), while `min_profile_size`'s
was large (1.615) — the model was leaning almost entirely on profile size,
not chemistry. An ablation confirmed it: size features alone accounted for
essentially the whole jump (0.7046 → 0.8713), fingerprint similarity alone
added almost nothing (0.7046 → 0.7064).

**Step 3 — size-matched negative sampling (`scripts/run_pair_link_classifier_sizematched_kfold.py`):
the honest number.** Profile size is a well-known confound in link-prediction
benchmarks generally ("degree bias") — a drug with a large documented
profile mechanically has more *chances* to show shared-protein overlap
with anything, and separately, well-studied drugs get more research
attention and more documented interactions, regardless of true mechanism.
Fixed with the standard remedy: sample negative pairs with each endpoint
drawn proportional to its degree within the sampled positives, so
negatives' size distribution matches positives' by construction (sanity
check: mean `min_profile_size` 4.32 positives vs. 3.30 negatives, much
closer than uniform-random sampling gives). Re-running the same 8-feature
model on these harder negatives:

| | ROC-AUC | precision | recall | F1 |
|---|---|---|---|---|
| uniform-random negatives (step 2) | 0.874 ± 0.003 | 0.875 | 0.663 | 0.754 |
| **size-matched negatives (honest)** | **0.650 ± 0.003** | 0.702 | 0.419 | 0.525 |

Learned coefficients flipped in an informative way once the size shortcut
was removed: `tanimoto_sim` went from ~0 to **0.149** (fingerprint
similarity carries real signal — it was just masked by the model
preferring the free size shortcut when both were available), `enzyme`
(0.545) and `target` (0.586) became nearly co-equal (previously enzyme
dominated at 1.91 vs. target's 0.66, itself likely partly a size artifact
since enzyme profiles correlate with overall profile size), `carrier`
stayed near-zero (0.033), and the size features correctly dropped to
~0 as the model learned they were no longer informative.

**Honest conclusion**: real signal survives — ROC-AUC 0.650 vs. 0.500
random is a consistent, non-trivial effect (std only ±0.003 across
folds) — but it's meaningfully smaller than either the naive baseline
(0.704) or the uncorrected rich-feature model (0.874) suggested. Worth
noting: even the *original* link-prediction benchmark's 0.704 ROC-AUC
almost certainly benefited from the same confound to some degree (larger
profiles mechanically create more overlap opportunities even absent real
interaction), so 0.65 may be closer to this project's true "clean"
ceiling for protein-overlap/structure-based link prediction than either
number reported before it. This is a well-documented general phenomenon
in link-prediction literature, not unique to this project — it's also
plausible (though unverified without re-running their exact evaluation)
that some of DeepDDI/SumGNN/medicX's headline 92-95% numbers benefit from
the same degree-bias effect, since they're evaluated the identical way
(random held-out edges in a graph where node degree is a well-established
strong predictor).

**Not yet deployed.** Given the meaningful gap between the flattering and
honest numbers, any integration into `scripts/mechanism_lookup.py` should
use the size-matched-trained model (or drop the size features from the
deployed feature set entirely, keeping only the 4 overlap counts +
tanimoto_sim, which carry the genuine signal) and quote the honest 0.650
ROC-AUC, not the inflated 0.874 — a decision deliberately left open
pending explicit sign-off, same as this project's established pattern for
every other validated-but-not-yet-shipped result.

**How the honest 0.650 actually compares to the published literature** —
clearly behind, by a real margin, not a rounding difference:

| | ROC-AUC / equivalent |
|---|---|
| **This project (debiased, size-matched)** | **0.650** |
| SumGNN (TWOSIDES, transductive) | ROC-AUC 0.949 |
| SumGNN (DrugBank, transductive) | accuracy 92.7% / F1 86.9 |
| medicX (KG embedding) | F1 95.2% |
| DeepDDI | accuracy 92.4% |

**Why the gap is expected, not alarming, and what closing it would
actually require**: this project's model is 4 overlap counts + 1
fingerprint-similarity number fed into a plain logistic regression — 5
learned weights, no embeddings, no representation learning. SumGNN and
medicX are deep models learning from a fundamentally richer signal: not
just "what does drug A touch, what does drug B touch" (this project's
entire feature set), but the *pattern* of which other drugs interact with
things structurally/relationally similar to A and B — collaborative-
filtering-style reasoning over a large auxiliary knowledge graph (SumGNN's
Hetionet: 33,765 nodes, 1.69M edges) plus the DDI graph's own structure.
Comparing a 5-parameter linear model against a graph neural network with a
knowledge base behind it is comparing genuinely different amounts of
modeling investment, not a fair architecture-controlled comparison.

**One honest caveat that could narrow the gap, currently unverified**: it
is unknown whether SumGNN/medicX corrected for the same degree-bias
confound this section's own ablation just found and fixed (matched
negative sampling is not universal practice in this literature). If they
didn't, some of their 0.87–0.95-level numbers could be inflated the same
way this project's own uncorrected 0.874 was — meaning the *true* gap
might be smaller than the raw numbers suggest. This can't be confirmed
without reproducing their exact evaluation protocol, which was explicitly
scoped out of this comparison (conceptual/literature comparison only, not
benchmark reproduction — see the earlier SumGNN comparison section above).

**What would actually close this gap**: the same knowledge-graph lever
already discussed and set aside for a different reason. The earlier
Hetionet discussion (see "Comparison to published DDI models" above)
argued AGAINST it for the mechanism gap-filler specifically, because a
genuinely novel/cold-start drug has no graph edges to leverage regardless
of how rich the graph is. That objection does **not** apply here — this
pair-classifier task is explicitly about drugs DrugBank already documents
individually, so a knowledge graph is architecturally the right tool for
*this* gap, unlike the other one. Not attempted: a real build (new
dataset, ID mapping to DrugBank IDs, GNN architecture), not a quick
follow-up on top of what exists.
