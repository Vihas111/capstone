"""
MechanismMLP: predicts a drug's likely mechanism-label profile (which
enzymes/targets/transporters it affects, with what action) from its
molecular fingerprint alone -- the gap-filler for drugs/pairs
scripts/mechanism_lookup.py has no documented DrugBank data for.

Deliberately small and heavily regularized: the trainable population here is
only ~3,880 drugs (see scripts/build_mechanism_labels.py), two orders of
magnitude smaller than the ADE-GNN track's ~172k pair-graphs, so a model
sized like models/rgcn_v2.py would badly overfit. Same
`prior_logits`-bias-init pattern as RGCNv2 -- this project has hit base-rate
collapse before (see CLAUDE.md), and the same mechanism-label sparsity
(median 2 positive labels/drug over ~422 classes) puts this task in the same
danger zone.
"""

import torch
import torch.nn as nn


class MechanismMLP(nn.Module):

    def __init__(
        self,
        in_dim=1024,
        hidden_dims=(256, 128),
        out_dim=422,
        dropout=0.4,
        prior_logits=None,
    ):
        """
        in_dim       : fingerprint bit-width (see scripts/build_fingerprints.py).
        hidden_dims  : sizes of the hidden layers, in order.
        out_dim      : number of mechanism labels (see
                       data/processed/mechanism_label_vocab.json).
        prior_logits : optional [out_dim] tensor of empirical per-class log-
                       odds. Initializes the final layer's bias to these
                       values instead of zero, so training starts already at
                       each class's base rate instead of having to learn it
                       from scratch on a small, sparse dataset -- same fix
                       as models/rgcn_v2.py's prior_logits param.
        """

        super().__init__()

        dims = [in_dim, *hidden_dims]
        layers = []

        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d_in, d_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        self.trunk = nn.Sequential(*layers)
        self.classifier = nn.Linear(dims[-1], out_dim)

        if prior_logits is not None:
            if prior_logits.shape != (out_dim,):
                raise ValueError(
                    f"prior_logits shape {tuple(prior_logits.shape)} doesn't "
                    f"match out_dim {out_dim}"
                )
            with torch.no_grad():
                self.classifier.bias.copy_(prior_logits)

    def forward(self, x):
        return self.classifier(self.trunk(x))
