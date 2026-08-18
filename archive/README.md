# Archive: superseded v1 (ADE-effect RGCN) pipeline

These are the pre-fix v1 files for the ADE-effect prediction track, kept for
reference/comparison — not for active use. The current version of this track
lives at the top level (`models/rgcn_v2.py`, `data_utils/shard_dataset_v2.py`,
`scripts/train_v2.py`, `scripts/evaluate_v2.py`, `scripts/tensorize_pair_graphs_v2.py`).

Confirmed bugs in the v1 files below, all fixed in v2:

- `models/rgcn_baseline.py` — hardcoded `nn.Embedding(5000, ...)` for every
  node type with `indices.clamp(max=4999)`. There are ~19,718 distinct drugs,
  so thousands silently collide onto the same embedding row.
- `scripts/create_splits.py` — splits shards with plain `random.shuffle` at
  the shard level, so the same drug can appear in both train and val/test,
  leaking identity and inflating validation metrics.
- `data_utils/shard_dataset.py` — an `IterableDataset` with no
  `get_worker_info()` handling, so under `num_workers>1` every worker
  iterates the full shard list — each graph is yielded N times per "epoch",
  unshuffled.
- `scripts/train_rgcn.py` / `scripts/train_full.py` — never load
  `compute_pos_weight.py`'s output, so `BCEWithLogitsLoss()` runs with no
  class-imbalance correction despite ~1:95 positive:negative imbalance.
- `scripts/evaluate_rgcn.py` / `scripts/evaluate_test_rgcn.py` — report only
  micro PR-AUC/ROC-AUC over ~4486 heavily-imbalanced classes, dominated by
  frequent classes; also still hardcoded to the buggy `RGCNBaseline`.

## Contents

- `scripts/` — `tensorize_pair_graphs.py`, `create_splits.py`, `train_rgcn.py`,
  `train_full.py`, `evaluate_rgcn.py`, `evaluate_test_rgcn.py`
- `models/rgcn_baseline.py`, `data_utils/shard_dataset.py`
- `data/tensorized/` — v1 tensorization (local per-graph position ids as node
  features, not real global ids — superseded by `data/tensorized_v2/`)
- `checkpoints/` — `best_rgcn.pt` (trained on the buggy v1 pipeline),
  `rgcn_smoke_test.pt` (a 125-step debug run, not a real trained model),
  `training_history.json`, `val_predictions.pt` (raw prediction dump from the
  old `evaluate_rgcn.py`)

**Note:** `archive/scripts/create_splits.py` defaults to globbing
`data/tensorized` — that path no longer exists at the top level (it moved to
`archive/data/tensorized`, right next to this script) — so running it as-is
without editing the path will find nothing. It's kept only for reference; the
live pipeline no longer uses it (see top-level `data/splits/*.txt`, already
regenerated against `tensorized_v2`, and top-level `CLAUDE.md`).

## Also archived: 5 v1 analysis scripts missed by the original archival pass (2026-08-18)

`scripts/predict_pair_from_dataset.py`, `scripts/compare_true_vs_pred.py`,
`scripts/analyze_label_distribution.py`, `scripts/error_analysis.py`,
`scripts/explain_prediction.py` — found while investigating
`data/tensorized_test/` (see CLAUDE.md section 6 item 7 / section 4c). All
five hardcode v1-era paths that no longer exist at the top level:
`checkpoints/best_rgcn.pt` (not `best_rgcn_v2.pt`), `data/tensorized/shard_*.pt`
/ `data/tensorized/ade_vocab.json` (not `tensorized_v2`), and
`checkpoints/val_predictions.pt` — all of which were archived here in the
original v1 cleanup pass, but these 5 scripts themselves were left behind in
the live `scripts/` directory, silently broken (would raise
`FileNotFoundError` if run) rather than actually archived alongside their
dependencies. Moved here now for consistency with what this README already
claimed was done. Not fixed to point at v2 paths — `scripts/evaluate_v2.py`
already covers the "how good is the current model" question this track
needs; these were v1-specific interpretation/debugging tools, not something
actively missing from the v2 workflow.

## Also archived: superseded mechanism-lookup prototype (2026-08-17)

`scripts/interaction_index.py` (`InteractionIndex`: per-drug enzyme/
transporter/target loaders + exact-name→ID resolution) and
`scripts/mechanism_engine.py` (`MechanismEngine`: pairwise-only enzyme/
transporter conflict detection with directional PK reasoning, e.g.
inhibitor+substrate → "may increase exposure"). Both were early, unfinished,
never-imported-elsewhere prototypes of what `scripts/mechanism_lookup.py`
now does properly (N-drug combinations, not just pairs; documented DrugBank
interactions; carrier overlap; JSON output; the downstream LLM prompt). Their
useful parts — exact-name resolution and the inhibitor/substrate/inducer PK
reasoning rules — were ported into `mechanism_lookup.py` (`resolve_drug`,
`infer_pk_effects`) before archiving these.
