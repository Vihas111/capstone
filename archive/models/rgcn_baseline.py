import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import HeteroConv, SAGEConv


ALL_NODE_TYPES = [
    "drug",
    "target",
    "enzyme",
    "transporter",
    "carrier",
    "pathway",
    "category",
]


class RGCNBaseline(nn.Module):

    def __init__(
        self,
        metadata,
        hidden_dim=64,
        out_dim=4486,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # --------------------------------------------------
        # Global node embeddings (not batch-dependent)
        # --------------------------------------------------
        self.node_embeddings = nn.ModuleDict(
            {
                node_type: nn.Embedding(
                    5000,
                    hidden_dim,
                )
                for node_type in ALL_NODE_TYPES
            }
        )

        # --------------------------------------------------
        # IMPORTANT:
        # Use explicit dimensions instead of (-1, -1)
        # to avoid lazy initialization problems.
        # --------------------------------------------------
        self.conv1 = HeteroConv(
            {
                edge_type: SAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                )
                for edge_type in metadata[1]
            },
            aggr="sum",
        )

        self.conv2 = HeteroConv(
            {
                edge_type: SAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                )
                for edge_type in metadata[1]
            },
            aggr="sum",
        )

        # --------------------------------------------------
        # Classification head
        # --------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):

        # --------------------------------------------------
        # Initial node embeddings
        # --------------------------------------------------
        x_dict = {}

        for node_type, x in data.x_dict.items():

            indices = x.squeeze(-1)
            indices = indices.clamp(max=4999)

            x_dict[node_type] = self.node_embeddings[
                node_type
            ](indices)

        # --------------------------------------------------
        # First message-passing layer
        # Preserve embeddings for node types that receive
        # no messages in the current batch.
        # --------------------------------------------------
        x_dict_new = self.conv1(
            x_dict,
            data.edge_index_dict,
        )

        for node_type in x_dict:

            if node_type in x_dict_new:
                x_dict[node_type] = F.relu(
                    x_dict_new[node_type]
                )

        # --------------------------------------------------
        # Second message-passing layer
        # --------------------------------------------------
        x_dict_new = self.conv2(
            x_dict,
            data.edge_index_dict,
        )

        for node_type in x_dict:

            if node_type in x_dict_new:
                x_dict[node_type] = F.relu(
                    x_dict_new[node_type]
                )

        # --------------------------------------------------
        # Every graph contains exactly two drug nodes.
        # Use their mean as the graph representation.
        # --------------------------------------------------
        drug_repr = x_dict["drug"]

        batch_size = data["drug"].ptr.numel() - 1

        drug_repr = drug_repr.view(
            batch_size,
            2,
            self.hidden_dim,
        )

        graph_repr = drug_repr.mean(dim=1)

        logits = self.classifier(
            graph_repr
        )

        return logits