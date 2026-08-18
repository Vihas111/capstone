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

3. **Pretrained molecular embeddings — condition now met (item 1 confirmed
   a representation problem, not a data-volume problem).** Self-training
   already tested "more data, same representation" and it didn't help,
   consistent with (not just "weakly suggestive of") this conclusion. Still
   the highest-effort item of the four, and item 2 is cheaper — try that
   first unless there's a reason to skip ahead. If pursued, evaluate on the
   existing k-fold splits so it's a fair comparison against the
   0.258 ± 0.013 baseline (and separately against per-kind baselines
   enzyme 0.143 ± 0.027 / transporter 0.154 ± 0.009 / target 0.310 ± 0.011),
   not a new single-split number that repeats this round's original
   mistake.

4. **PK/PD (now: transporter-specific) model splitting** — still not
   attempted. Lowest priority of the four: per-category pos_weight already
   tested the "make the loss pay more attention to the weak kind" idea via
   a cheaper mechanism and it didn't help, which is weak evidence a full
   separate model/head for transporter wouldn't help either (though not
   conclusive — a separate model changes capacity allocation, not just loss
   weighting).

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
