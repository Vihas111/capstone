"""
Fixes vs. models/rgcn_baseline.py:

1. Embedding-collision bug (CLAUDE.md §4 item iv -- the big one):
   The original does
       nn.Embedding(5000, hidden_dim)   for EVERY node type
       indices.clamp(max=4999)
   but CLAUDE.md states DrugBank has ~19,830 drugs. Any drug with a global id
   >= 5000 gets silently clamped to slot 4999 -- i.e. thousands of distinct
   drugs share one embedding row, and the model literally cannot tell them
   apart. This is almost certainly why the original model plateaus near a
   base-rate/majority-class prediction. Fixed here by taking real per-node-type
   vocab sizes (see scripts/build_vocab_sizes.py) instead of one hardcoded
   constant, and by raising loudly instead of silently clamping if something
   is still out of range.

2. Fragile "exactly two drug nodes" reshape:
   `drug_repr.view(batch_size, 2, hidden_dim)` will silently produce garbage
   (wrong reshape, no error) the moment a graph doesn't have exactly 2 drug
   nodes -- which is exactly the direction this project wants to go per
   CLAUDE.md's roadmap (KG-message-passing / N-drug extension). Fixed here
   using `data['drug'].batch` to pool per-graph explicitly, with an assertion
   that catches unexpected node counts instead of silently corrupting output.

3. Weak interaction modeling:
   Mean-pooling the two drug embeddings throws away *how* they interact.
   Added an optional bilinear head (CLAUDE.md's own next-step suggestion,
   also in project memory) that models drug_a^T W drug_b in addition to the
   pooled representation.

4. Added LayerNorm + a residual connection between conv layers, which is a
   normal fix for HeteroConv stacks like this one (helps gradient flow /
   avoids the 2nd layer's updates being dominated by initial noisy embeddings).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.utils import scatter


class RGCNv2(nn.Module):

    def __init__(
        self,
        metadata,
        vocab_sizes,
        hidden_dim=128,
        out_dim=4486,
        num_layers=2,
        dropout=0.2,
        head="bilinear",  # "mean" or "bilinear"
        prior_logits=None,
    ):
        """
        metadata     : (node_types, edge_types) as from HeteroData.metadata()
        vocab_sizes  : dict[node_type -> int], e.g. output of
                       scripts/build_vocab_sizes.py. MUST cover every node
                       type in metadata[0]. Sizing this correctly is the #1
                       fix over the baseline model.
        head         : "mean" (original behaviour, pooled mean of the two
                       drug embeddings) or "bilinear" (adds a learned pairwise
                       interaction term on top of the pooled representation).
        prior_logits : optional [out_dim] tensor of empirical per-class log-
                       odds (see scripts/build_label_priors.py). Initializes
                       the classifier's final-layer bias to these values
                       instead of zero, so training starts already at each
                       class's base rate -- addresses base-rate collapse
                       (model learns to ignore the input entirely) confirmed
                       via a direct prediction-variance check on this project's
                       real data, where trained-model outputs had ~1.0 cosine
                       similarity across different drug pairs.
        """

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.head_type = head

        node_types, edge_types = metadata

        missing = set(node_types) - set(vocab_sizes)
        if missing:
            raise ValueError(
                f"vocab_sizes is missing entries for node types: {missing}. "
                f"Run scripts/build_vocab_sizes.py to generate a complete file."
            )

        self.node_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(
                    vocab_sizes[node_type] + 1,  # +1 headroom, see forward()
                    hidden_dim,
                )
                for node_type in node_types
            }
        )
        self._vocab_sizes = dict(vocab_sizes)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):

            self.convs.append(
                HeteroConv(
                    {
                        edge_type: SAGEConv(
                            (hidden_dim, hidden_dim),
                            hidden_dim,
                        )
                        for edge_type in edge_types
                    },
                    aggr="sum",
                )
            )

            self.norms.append(
                nn.ModuleDict(
                    {
                        node_type: nn.LayerNorm(hidden_dim)
                        for node_type in node_types
                    }
                )
            )

        self.dropout = nn.Dropout(dropout)

        pooled_dim = hidden_dim

        if head == "bilinear":
            self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, hidden_dim)
            pooled_dim = hidden_dim * 2  # [mean_pool ; bilinear_interaction]
        elif head != "mean":
            raise ValueError(f"Unknown head type: {head!r}")

        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        if prior_logits is not None:
            final_layer = self.classifier[-1]
            if prior_logits.shape != (out_dim,):
                raise ValueError(
                    f"prior_logits shape {tuple(prior_logits.shape)} doesn't "
                    f"match out_dim {out_dim}"
                )
            with torch.no_grad():
                final_layer.bias.copy_(prior_logits)

    def _embed_inputs(self, x_dict):

        out = {}

        for node_type, x in x_dict.items():

            indices = x.squeeze(-1)

            max_idx = self._vocab_sizes[node_type]
            if int(indices.max()) > max_idx:
                # Loud failure instead of the original's silent clamp: a bad
                # vocab_sizes file should be obvious immediately, not hidden
                # as a training-quality mystery.
                raise ValueError(
                    f"Node type '{node_type}' has index {int(indices.max())} "
                    f"but vocab_sizes says max is {max_idx}. Re-run "
                    f"scripts/build_vocab_sizes.py against the current data."
                )

            out[node_type] = self.node_embeddings[node_type](indices)

        return out

    def forward(self, data):

        x_dict = self._embed_inputs(data.x_dict)

        for layer_idx in range(self.num_layers):

            x_dict_new = self.convs[layer_idx](
                x_dict,
                data.edge_index_dict,
            )

            for node_type in x_dict:

                if node_type in x_dict_new:

                    updated = self.norms[layer_idx][node_type](
                        x_dict_new[node_type]
                    )
                    updated = F.relu(updated)
                    updated = self.dropout(updated)

                    # Residual connection: node types that DO receive
                    # messages keep a footpath back to their embedding
                    # instead of fully overwriting it each layer.
                    x_dict[node_type] = x_dict[node_type] + updated

        drug_repr = x_dict["drug"]
        drug_batch = data["drug"].batch  # graph index per drug node

        num_graphs = int(drug_batch.max()) + 1
        counts = torch.bincount(drug_batch, minlength=num_graphs)

        if not torch.all(counts == 2):
            raise ValueError(
                "RGCNv2's pairwise head assumes exactly 2 drug nodes per "
                f"graph; got counts={counts.tolist()}. If you're extending "
                "to N-drug combinations, use a pooling head instead (see "
                "CLAUDE.md's N-drug roadmap note) rather than the bilinear "
                "pairwise head."
            )

        # Pooled (order-invariant) representation -- same as the baseline.
        mean_pool = scatter(drug_repr, drug_batch, dim=0, reduce="mean")

        if self.head_type == "mean":
            graph_repr = mean_pool
        else:
            # Recover the two per-graph drug embeddings in a stable order
            # (first occurrence, second occurrence). This relies on PyG's
            # Batch.from_data_list always laying out nodes contiguously by
            # graph (true in practice for DataLoader-produced batches), so
            # "first node of a graph" == "where batch id changes".
            is_first = torch.ones_like(drug_batch, dtype=torch.bool)
            is_first[1:] = drug_batch[1:] != drug_batch[:-1]
            is_second = ~is_first

            drug_a = drug_repr[~is_second]
            drug_b = drug_repr[is_second]

            interaction = self.bilinear(drug_a, drug_b)
            graph_repr = torch.cat([mean_pool, interaction], dim=-1)

        logits = self.classifier(graph_repr)

        return logits
