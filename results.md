# Pair-level link classifier: model comparison

Two models for predicting undocumented DrugBank drug-drug interactions
between already-known drugs (the "known drugs, new combinations" task).
See `findings/findings.md`'s "Pair-level link classifier" and later
sections for the full experimental history; this file is a focused
comparison of the two strongest results.

## Model 1: Spectral Embedding + Gradient-Boosted Trees (0.784 ROC-AUC)

**Implementation method**

This is a classic feature-engineering + tabular-ML pipeline, not a neural network. For a drug pair (A, B), it builds a 122-number feature vector from four sources, all computed *before* any labels are looked at:

1. **DrugBank features (8)** — shared enzyme/target/transporter/carrier counts, Tanimoto fingerprint similarity, and each drug's documented-profile size (already used in the earlier, honest-0.650 baseline).
2. **Direct Hetionet overlap (8)** — shared genes bound/up/down-regulated, shared side effects, shared disease indications, shared pharmacologic class, a direct "Hetionet says these are chemically similar" flag.
3. **2-hop mechanism features (8)** — gene-gene-mediated overlap ("A's gene interacts with B's gene," not just "same gene"), pathway-mediated overlap, and Adamic-Adar-weighted versions of the direct counts (down-weighting promiscuous hub proteins).
4. **Spectral graph embedding (98)** — the interesting part. Treat the *entire* Hetionet graph (47,031 nodes, all 24 relation types — genes, pathways, diseases, anatomy, everything) as one big adjacency matrix, symmetrically degree-normalize it, and run truncated SVD (`scipy.sparse.linalg.svds`, k=96) to get a 96-dimensional vector per node. This is a spectral/linear-algebra technique — no gradient descent, no labels involved, purely a compressed summary of "where does this node sit in the graph's overall structure." For a pair, we take cosine similarity, dot product, and the elementwise product of the two drugs' vectors.

All 122 features get fed into a `HistGradientBoostingClassifier` (sklearn's gradient-boosted trees), trained normally on ~100k sampled labeled pairs (50k documented DrugBank interactions as positives, 50k degree-matched negatives).

**Output method**

For any new pair, recompute the same 122 features (all deterministic lookups/arithmetic) and call `.predict_proba()` → a single probability in [0,1]. No randomness at inference. The model also exposes feature importances, so a prediction can be partly explained ("driven mostly by shared target overlap and graph-embedding similarity").

**Scalability**

- **Inference**: trivial — CPU-only, sub-millisecond per pair, no GPU anywhere in the path.
- **One-time setup cost**: building the graph adjacency structures and running the SVD takes ~10s and needs the ~12MB Hetionet file in memory. This is currently redone from scratch every script run — for production it should be cached/persisted rather than recomputed per query batch, but that's a small engineering fix, not a fundamental limitation.
- **Retraining trigger**: only needed if DrugBank's documented interactions change meaningfully, or if you want to refresh the Hetionet snapshot. Otherwise this is a train-once-and-freeze system.
- **Portability**: fully CPU-only, no CUDA dependency anywhere — works identically on both machines in this project's two-machine setup without any of the torch-version juggling CLAUDE.md documents for the training pipelines.
- **Population ceiling**: only works for drugs that are both in Hetionet's 1,552 compounds and in DrugBank's profile-bearing population (1,460 after intersecting) — same hard ceiling as Model 2, not a difference between them.

## Model 2: Task-Supervised Graph Neural Network (0.945 ROC-AUC)

**Implementation method**

This is fundamentally different: instead of computing fixed features and handing them to a separate classifier, the *representation itself* is learned, driven directly by the DDI labels.

- **Node embeddings**: every one of the 47,031 Hetionet nodes gets a randomly-initialized, *learnable* 128-dim vector (`nn.Embedding`) — there's no hand-designed feature here at all, just free parameters.
- **Graph convolution**: two `GCNConv` layers (torch_geometric) perform message passing — each node's vector gets mixed with its neighbors' vectors, twice, so after the two layers a node's final embedding reflects its 2-hop neighborhood, weighted by the real graph structure (all 4.5M directed edges, all relation types treated uniformly — this is a simplification versus a true relation-aware GNN like SumGNN's, which weights different edge types differently).
- **Decoder**: for a pair (A, B), take their final embeddings z_A, z_B, build a symmetric combination `[z_A*z_B, |z_A - z_B|, z_A + z_B]` (symmetric so the pair scores the same regardless of input order — drug pairs are unordered), and pass it through a small 3-layer MLP to a single logit.
- **Training**: everything — the 47k node embeddings, the GCN's weights, the decoder's weights — is trained *end-to-end* via backpropagation against binary cross-entropy on the sampled labeled pairs. Each epoch does one full forward pass over the *entire* graph (computing all 47k node embeddings together, since message passing is inherently a whole-graph operation), then reads off just the embeddings needed for the current batch of training pairs to compute the loss. This took up to 500 epochs per fold to converge (~13 minutes for a full 5-fold evaluation on this laptop's RTX 4060).

The reason this scores so much higher: because the loss reaches the node embeddings directly, a drug's embedding absorbs signal from *all* of its other training-fold interaction labels, not just generic structural position. The ablation performed to check this confirmed both ingredients matter — stripping out the graph message passing entirely (keeping only free-floating trained vectors) dropped the same fold from 0.951 to 0.874, so the actual Hetionet biology is doing real work on top of the task-supervision, not just riding along.

**Output method**

For a new pair, you need the trained node-embedding table (produced by one forward pass of the trained GCN over the whole graph) and the trained decoder. Look up A and B's rows in that table, run the decoder, sigmoid → probability. Deterministic once trained, but — importantly — a "new" prediction still conceptually depends on that shared whole-graph embedding table, not an independent per-drug computation.

**Scalability**

- **Inference**: should be cheap once trained — a single forward pass through a small 2-layer GCN plus a tiny MLP. CPU-only inference latency hasn't been explicitly benchmarked, so this should be confirmed before treating it as "definitely fine," but there's no obvious reason it'd be expensive.
- **Training/retraining cost**: real — GPU strongly preferred (~13 min on this laptop's RTX 4060 for the full evaluation; CPU-only training would likely be substantially slower, untested).
- **Retraining trigger — this is the real scalability liability**: the model is *transductive*, meaning its node-embedding table is indexed to one specific snapshot of Hetionet's node list. If a new drug is added, or DrugBank's interaction data changes, there's no "embed this one new node" shortcut — the whole model needs retraining from scratch. Model 1, by contrast, computes its features fresh at query time from raw graph structure, so it degrades more gracefully to graph updates (recompute features, no retraining required unless you want the classifier itself refreshed).
- **Storage**: the embedding table alone is 47,031 × 128 floats ≈ 24MB, plus small GCN/decoder weight files — needs to be versioned alongside the exact Hetionet snapshot it was trained on, since node indices are snapshot-specific.
- **Not yet built for deployment**: every number reported so far comes from 5 independently-retrained, throwaway fold-models used only for evaluation. Actually deploying this means a separate step: train one final model on the full pair set and persist its weights — not done yet.

## Side-by-side

| | Model 1 (spectral + HGB) | Model 2 (task-supervised GNN) |
|---|---|---|
| ROC-AUC | 0.784 | 0.945 |
| Needs GPU | No | For training, yes (practically) |
| Inference cost | Sub-ms, CPU | Cheap, likely CPU-fine (unverified) |
| Retrains when... | Rarely (features are frozen) | Whenever population/pairs change (transductive) |
| Portable across your two machines | Fully | Needs torch + torch_geometric on both (already in requirements.txt) |
| What's learned | Nothing — SVD is unsupervised, only the tree classifier is fit | Everything — node vectors, graph weights, and decoder, all jointly |
