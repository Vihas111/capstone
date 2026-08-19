"""
Loads Hetionet v1.0 (Himmelstein et al. 2017, eLife) and exposes per-drug-pair
features from it -- the knowledge-graph lever identified in
findings/findings.md as the one untried, architecturally-appropriate way to
close the gap between this project's pair-level link classifier (honest,
size-matched ROC-AUC 0.650 -- see scripts/run_pair_link_classifier_sizematched_kfold.py)
and published KG-based approaches like SumGNN/medicX, which both rely on a
large auxiliary knowledge graph the DrugBank-only overlap features don't have
access to.

Data: raw/hetnet TSV export, downloaded from github.com/hetio/hetionet
(hetnet/tsv/hetionet-v1.0-{nodes.tsv, edges.sif.gz}) into data/hetionet/.
Compound node IDs are DrugBank IDs directly (e.g. "Compound::DB00014"), so no
separate ID-mapping step is needed -- confirmed by inspection before writing
this loader.

Coverage note (important, checked before use): Hetionet only covers 1,552
compounds, vs. this project's 10,192-drug DrugBank profile population -- a
much smaller population than the existing link-classifier scripts use. Any
comparison against the 0.650 baseline MUST be re-run on the identical
restricted population, not compared against the old number directly, or the
comparison would confound "does the KG help" with "is this an easier/harder
population." scripts/run_pair_link_classifier_hetionet_kfold.py does this.

Metaedges used (Compound-anchored, chosen for plausible DDI relevance):
  CbG   Compound-binds-Gene            (ChEMBL/BindingDB binding data --
                                         broader than DrugBank's own curated
                                         target/enzyme/transporter actions)
  CuG   Compound-upregulates-Gene
  CdG   Compound-downregulates-Gene
  CcSE  Compound-causes-SideEffect     (SIDER single-drug side effects --
                                         shared side-effect profile as a
                                         proxy for shared mechanism; this is
                                         single-drug data, not drug-PAIR
                                         outcome data, so it's not the same
                                         signal as the TWOSIDES labels the
                                         RGCN track predicts -- no leakage
                                         into the thing being predicted here)
  CtD   Compound-treats-Disease
  CpD   Compound-palliates-Disease
  PCiC  PharmacologicClass-includes-Compound (reversed source/target in the
                                         raw file -- handled below)
  CrC   Compound-resembles-Compound    (Hetionet's own chemical-similarity
                                         edge, direct compound-compound, not
                                         neighbor-based like the others)
"""

import gzip
from collections import defaultdict
from pathlib import Path

HETIO_DIR = Path("data/hetionet")
NODES_PATH = HETIO_DIR / "hetionet-v1.0-nodes.tsv"
EDGES_PATH = HETIO_DIR / "hetionet-v1.0-edges.sif.gz"

NEIGHBOR_METAEDGES = ("CbG", "CuG", "CdG", "CcSE", "CtD", "CpD")
GENE_TOUCHING_METAEDGES = ("CbG", "CuG", "CdG")
GENE_GENE_METAEDGES = ("GiG", "GcG", "Gr>G")  # interacts, covaries, regulates (direction ignored, symmetrized)
HETIO_FEATURE_NAMES = [f"{m.lower()}_shared" for m in NEIGHBOR_METAEDGES] + [
    "pharm_class_shared",
    "resembles_direct",
]
HETIO_2ND_ORDER_FEATURE_NAMES = ["gene_gene_mediated_count", "pathway_mediated_shared"]
HETIO_ADAMIC_ADAR_FEATURE_NAMES = [f"{m.lower()}_adamic_adar" for m in NEIGHBOR_METAEDGES]


def _open_edges():
    if EDGES_PATH.exists():
        return open(EDGES_PATH, "rb")
    gz_path = EDGES_PATH.with_suffix(EDGES_PATH.suffix + ".gz") if EDGES_PATH.suffix != ".gz" else EDGES_PATH
    return gzip.open(gz_path, "rt")


def build_full_graph_embeddings(k=64, seed=42):
    """Spectral (truncated-SVD) node embeddings over the WHOLE Hetionet
    graph -- all 47,031 nodes, all 24 edge types, not just the
    Compound-anchored ones the other features above use. This is the
    "reason globally over the graph" lever the hand-counted local-overlap
    features (direct shared-neighbor counts, Adamic-Adar, gene/pathway
    2-hop) structurally can't provide: those only ever look 1-2 hops out
    from each drug, while this captures each node's position in the
    graph's overall community/connectivity structure -- the same kind of
    signal a GNN or KG-embedding (what SumGNN/medicX actually use) would
    learn, just via a much cheaper, dependency-light method (no
    gensim/node2vec available in this project's environment -- truncated
    SVD of the adjacency matrix is a well-established, theoretically
    grounded approximation to what DeepWalk/node2vec converge to, e.g. the
    NetMF equivalence result).

    No leakage risk: Hetionet does not contain DrugBank's own DDI edges at
    all (it has no drug-drug interaction edge type), so nothing about the
    thing being predicted here is baked into this embedding.

    Returns {node_id: np.ndarray[k]} for every node in the graph (not just
    compounds -- callers slice out what they need).
    """

    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import svds

    node_ids = []
    with open(NODES_PATH) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            node_ids.append(parts[0])
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    n = len(node_ids)

    rows, cols = [], []
    gz_path = HETIO_DIR / "hetionet-v1.0-edges.sif.gz"
    plain_path = HETIO_DIR / "hetionet-v1.0-edges.sif"
    opener = (lambda: open(plain_path)) if plain_path.exists() else (lambda: gzip.open(gz_path, "rt"))

    with opener() as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            source, _metaedge, target = parts
            si, ti = node_index.get(source), node_index.get(target)
            if si is None or ti is None:
                continue
            rows.append(si)
            cols.append(ti)
            rows.append(ti)
            cols.append(si)

    data = np.ones(len(rows), dtype=np.float64)
    adj = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()

    # Symmetric degree normalization (D^-1/2 A D^-1/2) -- standard spectral
    # embedding preprocessing, prevents high-degree hub nodes (e.g. genes
    # bound by hundreds of compounds) from dominating the singular vectors.
    degree = np.asarray(adj.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(degree)
    from scipy.sparse import diags
    d_mat = diags(d_inv_sqrt)
    adj_norm = d_mat @ adj @ d_mat

    u, s, _vt = svds(adj_norm, k=k, random_state=seed)
    embeddings = u * s  # [n, k]

    return {nid: embeddings[i] for nid, i in node_index.items()}


def embedding_pair_features(embeddings, a, b, k=64, with_elementwise=False):
    """[cosine_sim, dot_product] (+ optionally the k-dim elementwise
    product, when with_elementwise=True) between two Compound node
    embeddings. The elementwise product lets a tree model exploit
    individual embedding dimensions directly instead of only the single
    linear (dot) or angular (cosine) summary of them."""

    import numpy as np

    ea = embeddings.get(f"Compound::{a}")
    eb = embeddings.get(f"Compound::{b}")
    if ea is None or eb is None:
        base = [0.0, 0.0]
        return base + [0.0] * k if with_elementwise else base

    dot = float(np.dot(ea, eb))
    norm_a = float(np.linalg.norm(ea))
    norm_b = float(np.linalg.norm(eb))
    cos = dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0
    base = [cos, dot]
    if not with_elementwise:
        return base
    return base + (ea * eb).tolist()


def build_pyg_graph():
    """Returns (node_index, edge_index) for the whole Hetionet graph as a
    PyTorch tensor, for GNN-based (task-supervised) embedding -- the
    natural escalation past the unsupervised spectral embedding above:
    that one is frozen and never sees a single DDI label, this one lets a
    GCN's node embeddings be trained end-to-end against the actual
    link-prediction task, same as what SumGNN/medicX do (architecturally),
    just far simpler (no relation-aware message passing, no external KG
    embedding pretraining).
      node_index[node_id_str] -> int
      edge_index -> torch.LongTensor[2, 2*n_edges] (both directions, no
        self-loops -- GCNConv adds those internally)
    """

    import torch

    node_ids = []
    with open(NODES_PATH) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            node_ids.append(parts[0])
    node_index = {nid: i for i, nid in enumerate(node_ids)}

    rows, cols = [], []
    gz_path = HETIO_DIR / "hetionet-v1.0-edges.sif.gz"
    plain_path = HETIO_DIR / "hetionet-v1.0-edges.sif"
    opener = (lambda: open(plain_path)) if plain_path.exists() else (lambda: gzip.open(gz_path, "rt"))

    with opener() as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            source, _metaedge, target = parts
            si, ti = node_index.get(source), node_index.get(target)
            if si is None or ti is None:
                continue
            rows.append(si)
            cols.append(ti)
            rows.append(ti)
            cols.append(si)

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    return node_index, edge_index


def load_hetionet_compounds():
    """Set of DrugBank IDs that appear as Compound nodes in Hetionet."""
    compounds = set()
    with open(NODES_PATH) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            node_id, _name, kind = parts
            if kind == "Compound":
                compounds.add(node_id.split("::", 1)[1])
    return compounds


def load_hetionet_graph():
    """Returns (neighbor_sets, pharm_class_by_drug, resembles_pairs):
      neighbor_sets[metaedge][drug_id] -> set(neighbor node id string)
      pharm_class_by_drug[drug_id] -> set(pharmacologic class id)
      resembles_pairs -> set(frozenset((drug_a, drug_b))) from CrC edges
    """

    neighbor_sets = {m: defaultdict(set) for m in NEIGHBOR_METAEDGES}
    pharm_class_by_drug = defaultdict(set)
    resembles_pairs = set()

    gz_path = HETIO_DIR / "hetionet-v1.0-edges.sif.gz"
    plain_path = HETIO_DIR / "hetionet-v1.0-edges.sif"
    opener = (lambda: open(plain_path)) if plain_path.exists() else (lambda: gzip.open(gz_path, "rt"))

    with opener() as f:
        next(f)  # header: source, metaedge, target
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            source, metaedge, target = parts

            if metaedge in NEIGHBOR_METAEDGES:
                drug_id = source.split("::", 1)[1]
                neighbor_sets[metaedge][drug_id].add(target)

            elif metaedge == "PCiC":
                # source is Pharmacologic Class, target is Compound.
                pc_id = source.split("::", 1)[1]
                drug_id = target.split("::", 1)[1]
                pharm_class_by_drug[drug_id].add(pc_id)

            elif metaedge == "CrC":
                a = source.split("::", 1)[1]
                b = target.split("::", 1)[1]
                resembles_pairs.add(frozenset((a, b)))

    return neighbor_sets, pharm_class_by_drug, resembles_pairs


def build_node_degree(neighbor_sets):
    """node_degree[metaedge][node_id] -> number of distinct drugs touching
    that node. Feeds Adamic-Adar weighting below: raw shared-neighbor counts
    treat a rare, specific shared protein the same as a promiscuous hub one
    (e.g. a kinase half of ChEMBL binds) -- Adamic-Adar down-weights hubs,
    a standard, usually stronger link-prediction feature than raw counts."""

    node_degree = {}
    for metaedge, by_drug in neighbor_sets.items():
        counts = defaultdict(int)
        for nodes in by_drug.values():
            for n in nodes:
                counts[n] += 1
        node_degree[metaedge] = counts
    return node_degree


def adamic_adar_pair_features(neighbor_sets, node_degree, a, b):
    """[len(NEIGHBOR_METAEDGES)] Adamic-Adar score per metaedge for one pair:
    sum over shared neighbor nodes n of 1 / ln(degree(n)), degree(n) >= 2
    clamped so a node touched by only 1 drug (ln(1)=0, undefined) doesn't
    divide by zero -- such a node can't be shared by two different drugs
    anyway, so this only guards a defensive edge case."""

    import math

    feats = []
    for m in NEIGHBOR_METAEDGES:
        sa = neighbor_sets[m].get(a, set())
        sb = neighbor_sets[m].get(b, set())
        shared = sa & sb
        degrees = node_degree[m]
        score = sum(1.0 / math.log(max(degrees.get(n, 2), 2)) for n in shared)
        feats.append(score)
    return feats


def load_gene_graph():
    """Returns (gene_gene_adj, gene_pathway_adj) for the 2nd-order,
    mechanism/pathway-mediated features below. Most drug pairs have ZERO
    direct shared-neighbor overlap (see findings.md) -- these 2nd-order
    features capture "A's gene interacts with B's gene" and "A's gene and
    B's gene share a pathway" without requiring the exact same protein,
    which is a much less sparse signal.
      gene_gene_adj[gene_id] -> set(related gene id), symmetrized over
        GiG (interacts), GcG (covaries), Gr>G (regulates, direction dropped)
      gene_pathway_adj[gene_id] -> set(pathway id), from GpPW
    """

    gene_gene_adj = defaultdict(set)
    gene_pathway_adj = defaultdict(set)

    gz_path = HETIO_DIR / "hetionet-v1.0-edges.sif.gz"
    plain_path = HETIO_DIR / "hetionet-v1.0-edges.sif"
    opener = (lambda: open(plain_path)) if plain_path.exists() else (lambda: gzip.open(gz_path, "rt"))

    with opener() as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            source, metaedge, target = parts

            if metaedge in GENE_GENE_METAEDGES:
                # Keep the full "Gene::<id>" node id (not stripped) so these
                # keys match touched_genes' ids, which come from
                # neighbor_sets[CbG/CuG/CdG] and are stored with their
                # "Gene::" prefix intact -- a prior version of this function
                # stripped the prefix here, which silently zeroed every
                # gene_gene_mediated_count feature (namespace mismatch, no
                # error, just an always-empty intersection).
                gene_gene_adj[source].add(target)
                gene_gene_adj[target].add(source)

            elif metaedge == "GpPW":
                gene = source  # keep "Gene::<id>" prefix, same reason as above
                pw = target.split("::", 1)[1]
                gene_pathway_adj[gene].add(pw)

    return gene_gene_adj, gene_pathway_adj


def build_touched_genes_and_pathways(neighbor_sets, gene_pathway_adj):
    """Per-drug: touched_genes (union of CbG/CuG/CdG neighbor genes) and
    pathways (union of gene_pathway_adj over touched_genes). Precomputed
    once per drug so per-pair feature lookups are cheap set operations."""

    touched_genes = defaultdict(set)
    for m in GENE_TOUCHING_METAEDGES:
        for drug, genes in neighbor_sets[m].items():
            touched_genes[drug] |= genes

    pathways_by_drug = {}
    for drug, genes in touched_genes.items():
        pws = set()
        for g in genes:
            pws |= gene_pathway_adj.get(g, set())
        pathways_by_drug[drug] = pws

    return touched_genes, pathways_by_drug


def hetionet_2nd_order_pair_features(touched_genes, pathways_by_drug, gene_gene_adj, a, b):
    """[gene_gene_mediated_count, pathway_mediated_shared] for one pair."""

    genes_a = touched_genes.get(a, set())
    genes_b = touched_genes.get(b, set())

    # True metapath count Compound-Gene-Gene-Compound: for each gene A
    # touches, how many of B's genes are directly related to it.
    gene_gene_count = 0
    if genes_a and genes_b:
        for g1 in genes_a:
            gene_gene_count += len(gene_gene_adj.get(g1, set()) & genes_b)

    pathways_a = pathways_by_drug.get(a, set())
    pathways_b = pathways_by_drug.get(b, set())
    pathway_shared = len(pathways_a & pathways_b)

    return [gene_gene_count, pathway_shared]


def hetionet_pair_features(neighbor_sets, pharm_class_by_drug, resembles_pairs, a, b):
    """[len(HETIO_FEATURE_NAMES)] feature vector for one drug pair."""

    feats = []
    for m in NEIGHBOR_METAEDGES:
        sa = neighbor_sets[m].get(a, set())
        sb = neighbor_sets[m].get(b, set())
        feats.append(len(sa & sb))

    pca = pharm_class_by_drug.get(a, set())
    pcb = pharm_class_by_drug.get(b, set())
    feats.append(len(pca & pcb))

    feats.append(1.0 if frozenset((a, b)) in resembles_pairs else 0.0)

    return feats
