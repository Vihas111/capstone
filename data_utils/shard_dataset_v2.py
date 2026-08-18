"""
Fixes vs. data_utils/shard_dataset.py:

1. Multi-worker duplication bug (CLAUDE.md §4 item v):
   The original __iter__ has no idea it might be running inside one of several
   DataLoader workers. With num_workers=8 (as used in train_full.py), PyTorch
   forks 8 copies of this dataset, and EVERY copy iterates over the FULL shard
   list from the start. Net effect: every graph is yielded 8x per "epoch", the
   loss/metrics you see are silently computed over an 8x-inflated, non-shuffled,
   perfectly-repeated stream. This is fixed here by partitioning self.shards
   across workers using torch.utils.data.get_worker_info().

2. No shuffling at all:
   Because it's an IterableDataset, DataLoader(shuffle=True) is not supported
   and the original code has no substitute. Every epoch sees shards, and graphs
   within a shard, in exactly the same order. Fixed here with:
     - shard-order shuffling per epoch (seeded, deterministic across a run)
     - in-memory shuffling of the graphs within each loaded shard

3. Optional cold-start-by-drug filtering:
   Pass `drug_split` (the dict produced by scripts/build_drug_split.py) and a
   `split_name` ("train"/"val"/"test") to only yield graphs where BOTH drug
   node ids fall in that split's held-out drug set. This is what actually
   enforces "hold out entire drugs, not just pairs" (see CLAUDE.md's cold-start
   principle) -- create_splits.py only randomizes at the shard level, which
   does nothing to stop the same drug appearing in train and val.

Everything else (reverse-edge construction) is preserved from the original.
"""

import random
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info


# ------------------------------------------------------------------------
# Relation collapsing: DrugBank's raw pharmacological action strings (e.g.
# "inhibitor", "weak_inhibitor", "inhibitory_allosteric_modulator",
# "partial_agonist", "neutralizer", ...) produce ~170 near-unique edge
# relation types once tensorized. RGCNv2/RGCNBaseline builds one SAGEConv
# PER EDGE TYPE PER LAYER, so ~170 relations x 2 layers is ~340 separate
# weight matrices -- most backed by only a handful of graphs each. That is
# almost certainly a major contributor to the overfitting seen in training
# (huge relation-specific capacity, thin per-relation data), and matches
# CLAUDE.md's own note that HeteroConv dispatching ~160 relation-typed
# convs was a performance problem too.
#
# This is a DOMAIN judgment call, not a mechanical bugfix -- review/adjust
# the grouping below against real pharmacology before trusting it blindly.
# Unmapped relations pass through unchanged (so nothing is silently dropped
# if a name isn't in this table).
# ------------------------------------------------------------------------
RELATION_GROUPS = {
    # inhibitory-direction actions
    "inhibitor": "inhibits", "weak_inhibitor": "inhibits",
    "inhibitory_allosteric_modulator": "inhibits", "blocker": "inhibits",
    "downregulator": "inhibits", "suppressor": "inhibits",
    "disruptor": "inhibits", "degradation": "inhibits",
    "inhibition_of_synthesis": "inhibits", "negative_modulator": "inhibits",
    "inverse_agonist": "inhibits", "antagonist": "inhibits",
    "neutralizer": "inhibits",

    # activating-direction actions
    "inducer": "activates", "activator": "activates", "agonist": "activates",
    "partial_agonist": "activates", "potentiator": "activates",
    "stimulator": "activates", "upregulator": "activates",
    "positive_modulator": "activates", "positive_allosteric_modulator": "activates",
    "cofactor": "activates",

    # generic binding/association, direction-neutral
    "substrate": "interacts_with", "ligand": "interacts_with",
    "binder": "interacts_with", "binding": "interacts_with",
    "targets": "interacts_with", "adduct": "interacts_with",
    "antibody": "interacts_with", "allosteric_modulator": "interacts_with",
    "modulator": "interacts_with", "regulator": "interacts_with",
    "multitarget": "interacts_with", "product_of": "interacts_with",
    "intercalation": "interacts_with", "stabilization": "interacts_with",
    "cleavage": "interacts_with", "transporter": "interacts_with",
    "carrier_association": "interacts_with", "transported_by": "interacts_with",
    "associated_with": "interacts_with",

    "other": "other", "other/unknown": "other", "unknown": "other",

    "belongs_to": "belongs_to",
    "participates_in": "participates_in",
}


def group_relation(rel):
    """Collapse a raw relation name, stripping any 'rev_' prefix first so
    both directions of a relation map to the SAME grouped name (reverse
    edges get their own 'rev_' prefix re-applied by add_reverse_edges,
    which runs AFTER this grouping)."""

    if rel.startswith("rev_"):
        base = rel[len("rev_"):]
        return "rev_" + RELATION_GROUPS.get(base, base)

    return RELATION_GROUPS.get(rel, rel)


def apply_relation_grouping(data):
    """Rewrites a HeteroData's edge_types, merging any that collapse onto
    the same (src, grouped_rel, dst) key by concatenating their edge_index
    tensors. Must run BEFORE add_reverse_edges (reverse edges are built off
    already-grouped relation names, so both directions stay consistent)."""

    merged = {}

    for src, rel, dst in list(data.edge_types):

        grouped_rel = group_relation(rel)
        key = (src, grouped_rel, dst)

        edge_index = data[(src, rel, dst)].edge_index

        if key in merged:
            merged[key] = torch.cat([merged[key], edge_index], dim=1)
        else:
            merged[key] = edge_index

    for src, rel, dst in list(data.edge_types):
        del data[(src, rel, dst)]

    for (src, rel, dst), edge_index in merged.items():
        data[(src, rel, dst)].edge_index = edge_index

    return data


def add_reverse_edges(data):
    """Unchanged from the original -- adds a reverse relation for every
    existing edge type so message passing can flow in both directions."""

    edge_types = list(data.edge_types)

    for src, rel, dst in edge_types:

        edge_index = data[(src, rel, dst)].edge_index
        rev_rel = f"rev_{rel}"
        rev_edge_index = edge_index.flip(0)
        data[(dst, rev_rel, src)].edge_index = rev_edge_index

    return data


def _drug_ids_in_graph(graph):
    """Extract the raw (pre-embedding) global drug node ids from a graph's
    x_dict, matching the indexing scheme RGCNBaseline/RGCNv2 expect."""

    x = graph["drug"].x
    return [int(i) for i in x.squeeze(-1).tolist()]


class ShardDatasetV2(IterableDataset):

    def __init__(
        self,
        split_file,
        drug_split=None,
        split_name=None,
        shuffle=False,
        seed=0,
        group_relations=True,
    ):
        """
        split_file : path to a text file listing shard .pt paths (one per line)
        drug_split : optional dict like {"train": set(...), "val": set(...), "test": set(...)}
                     as produced by scripts/build_drug_split.py
        split_name : which key of drug_split to enforce ("train"/"val"/"test").
                     Required if drug_split is given.
        shuffle    : shuffle shard order each epoch, and shuffle graphs within
                     each shard after loading. Should be True for training,
                     False for val/test (repeatability).
        seed       : base seed; combined with an internal epoch counter so
                     repeated calls to __iter__ (= new epochs) reshuffle.
        group_relations : collapse DrugBank's ~170 near-unique action-string
                     relations into a small set of grouped types (see
                     RELATION_GROUPS above) before adding reverse edges.
                     IMPORTANT: whatever value you use here must exactly
                     match what was used when scripts/build_vocab_sizes.py
                     / metadata scans were run, or the model's HeteroConv
                     dict (built from metadata) won't line up with what
                     these batches actually contain.
        """

        with open(split_file) as f:
            self.shards = [
                Path(line.strip())
                for line in f
                if line.strip()
            ]

        if drug_split is not None and split_name is None:
            raise ValueError(
                "split_name is required when drug_split is provided"
            )

        self.drug_split = drug_split
        self.split_name = split_name
        self.allowed_drugs = (
            set(drug_split[split_name]) if drug_split is not None else None
        )

        self.shuffle = shuffle
        self.seed = seed
        self.group_relations = group_relations
        self._epoch = 0

    def _keep_graph(self, graph):

        if self.allowed_drugs is None:
            return True

        drug_ids = _drug_ids_in_graph(graph)

        return all(d in self.allowed_drugs for d in drug_ids)

    def __iter__(self):

        worker_info = get_worker_info()

        shards = list(self.shards)

        # Reshuffle shard order deterministically per-epoch, per-run.
        if self.shuffle:
            rng = random.Random(self.seed + self._epoch)
            rng.shuffle(shards)

        self._epoch += 1

        # --------------------------------------------------------------
        # THE ACTUAL FIX for the 8x-duplication bug: give each worker a
        # disjoint stride slice of the (already-shuffled) shard list, so
        # the union across all workers covers every shard exactly once.
        # --------------------------------------------------------------
        if worker_info is not None:
            shards = shards[worker_info.id :: worker_info.num_workers]

        for shard_path in shards:

            graphs = torch.load(
                shard_path,
                weights_only=False,
            )

            if self.shuffle:
                rng.shuffle(graphs)

            for graph in graphs:

                if not self._keep_graph(graph):
                    continue

                if self.group_relations:
                    graph = apply_relation_grouping(graph)

                yield add_reverse_edges(graph)
