"""
Deterministic mechanism-level lookup for a combination of drugs (polypharmacy).

Given N (>=2) DrugBank IDs and/or exact drug names, reports what DrugBank documents about each drug
individually (enzymes/targets/transporters/carriers it acts on, with actions
like inhibitor/substrate/inducer) plus, for every pair in the combination:

  - any DOCUMENTED interaction DrugBank's own interaction-description field
    states directly (drugbank_interactions.jsonl)
  - MECHANISTIC OVERLAPS: shared enzymes/targets/transporters between the two
    drugs, computed by intersecting their individual profiles. This is a
    separate, clearly-labeled signal -- it is NOT a documented interaction,
    just "these two drugs touch the same protein" (e.g. both inhibit
    CYP3A4), which is a plausible PK/PD interaction mechanism even when no
    DrugBank interaction text exists for that pair yet.

Every fact here comes directly from DrugBank tables already extracted by
extract_biomedical_features.py -- no ML, no LLM. That's what makes it
reliable for documented drugs: this is retrieval, not prediction. It will
say nothing about drug pairs/mechanisms DrugBank doesn't document (that's a
separate, future ML gap-filler problem -- see CLAUDE.md).

Usage (from the Capstone/ directory):
    python scripts/mechanism_lookup.py DB00682 DB00945 --pretty
    python scripts/mechanism_lookup.py warfarin "acetylsalicylic acid" DB01254 --out report.json

Name resolution is EXACT-MATCH only (case-insensitive) against DrugBank's
primary name in drugs.jsonl -- no synonym/brand-name table exists in this
repo yet (see CLAUDE.md). "Warfarin" resolves; "Coumadin" (a brand name)
won't. When in doubt, pass the DrugBank ID directly.

Pass --predict to ALSO run the structure-based ML gap-filler for every
input drug, predicting likely enzyme/target/transporter labels from the
drug's fingerprint alone. By default this blends the linear model
(models/mechanism_mlp.py, checkpoints/mechanism_predictor.pt) with a
Tanimoto-similarity k-NN vote over the main train split
(scripts/mechanism_knn_transfer.py) -- the ensemble beats the linear model
alone in 5/5 k-fold folds (findings/findings.md's "Item 2 result"); pass
--predict-no-knn-blend for the plain linear model instead. This is a
genuinely different kind of claim than everything else in this file -- a
MODEL GUESS, not a DrugBank fact -- so it is always kept in separate,
clearly-labeled fields (predicted_enzymes vs enzymes,
predicted_mechanistic_overlaps vs mechanistic_overlaps) and never merged
into the documented lists. Without --predict, output is byte-for-byte
identical to before this flag existed. torch/rdkit/models.mechanism_mlp are
imported lazily (only when --predict is passed) so plain lookups don't need
them installed.
"""

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import orjson

sys.path.append(str(Path(__file__).resolve().parent.parent))


DATA_DIR = Path("data/processed")
DEFAULT_PREDICT_CHECKPOINT = "checkpoints/mechanism_predictor.pt"
DEFAULT_PREDICT_VOCAB = "data/processed/mechanism_label_vocab.json"
DEFAULT_PREDICT_THRESHOLDS = "data/processed/mechanism_label_thresholds.json"
DEFAULT_PREDICT_THRESHOLDS_BLEND = "data/processed/mechanism_label_thresholds_blend.json"
DEFAULT_PREDICT_HIDDEN_DIMS = []  # [] = linear model, matches the deployed
# checkpoint -- empirically beat a (256,128) MLP on held-out test macro/micro
# -AP (see CLAUDE.md section 3b / scripts/evaluate_mechanism_predictor.py).
DEFAULT_PREDICT_FP_RADIUS = 2
DEFAULT_PREDICT_FP_NBITS = 1024
DEFAULT_PREDICT_FP_CHIRALITY = True  # must match scripts/build_fingerprints.py

# k-NN/linear ensemble (findings/findings.md "Item 2 result", 2026-08-18):
# beats the linear model alone in 5/5 k-fold folds (macro-AP 0.2927+-0.0156
# vs. 0.2581+-0.0137), so it's the --predict default. Neighbor pool is the
# main train split ONLY (never val/test), same population the linear model
# itself was trained on -- see scripts/tune_mechanism_blend_thresholds.py.
DEFAULT_KNN_K = 40  # matches scripts/run_mechanism_knn_ensemble_kfold.py's fold-0 val-sweep choice
DEFAULT_KNN_FINGERPRINTS = "data/processed/drug_fingerprints.pt"
DEFAULT_KNN_DATASET = "data/processed/mechanism_dataset.pt"
DEFAULT_KNN_SPLIT = "data/processed/mechanism_drug_split.json"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "drug_ids", nargs="+",
        help="2 or more DrugBank IDs and/or exact drug names, e.g. DB00682 warfarin",
    )
    p.add_argument("--pretty", action="store_true", help="Pretty-print JSON to stdout")
    p.add_argument("--out", default=None, help="Also save the JSON report to this path")
    p.add_argument(
        "--predict", action="store_true",
        help="Also run the structure-based ML gap-filler for every input drug "
        "(models/mechanism_mlp.py). Adds predicted_enzymes/predicted_targets/"
        "predicted_transporters + predicted_mechanistic_overlaps, kept "
        "separate from documented data. Without this flag, output is "
        "byte-for-byte identical to before --predict existed.",
    )
    p.add_argument("--predict-checkpoint", default=DEFAULT_PREDICT_CHECKPOINT)
    p.add_argument("--predict-vocab", default=DEFAULT_PREDICT_VOCAB)
    p.add_argument("--predict-hidden-dims", type=int, nargs="*",
                    default=DEFAULT_PREDICT_HIDDEN_DIMS)
    p.add_argument(
        "--predict-thresholds-path", default=None,
        help="Per-label thresholds tuned by scripts/tune_mechanism_thresholds.py / "
        "scripts/tune_mechanism_blend_thresholds.py (F1-optimal on the val split, "
        "with a support-gated per-kind fallback for labels too sparse to tune "
        "individually -- see scripts/mechanism_thresholds.py). Defaults to the "
        f"blend-tuned file ({DEFAULT_PREDICT_THRESHOLDS_BLEND}) unless "
        "--predict-no-knn-blend is passed, in which case it defaults to the "
        f"linear-only-tuned file ({DEFAULT_PREDICT_THRESHOLDS}). Falls back to "
        "the flat --predict-threshold if the resolved file is missing.",
    )
    p.add_argument(
        "--predict-threshold", type=float, default=None,
        help="Explicit flat threshold override for EVERY label, bypassing the "
        "tuned per-label thresholds file entirely. Also the fallback value if "
        "--predict-thresholds-path is missing.",
    )
    p.add_argument(
        "--predict-no-knn-blend", action="store_true",
        help="Use the plain linear model alone instead of the k-NN/linear "
        "ensemble blend. The blend is the default since findings/findings.md's "
        "Item 2 result showed it beats the linear model alone in 5/5 k-fold "
        "folds (macro-AP 0.2927+-0.0156 vs 0.2581+-0.0137) -- see CLAUDE.md "
        "section 6 item 4. This flag also switches the default thresholds "
        "file back to the linear-only-tuned one.",
    )
    p.add_argument("--predict-knn-k", type=int, default=DEFAULT_KNN_K)
    p.add_argument("--predict-knn-fingerprints", default=DEFAULT_KNN_FINGERPRINTS)
    p.add_argument("--predict-knn-dataset", default=DEFAULT_KNN_DATASET)
    p.add_argument("--predict-knn-split", default=DEFAULT_KNN_SPLIT)
    return p.parse_args()


def load_jsonl(path):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def load_drug_index():
    """Returns (id_to_name, name_to_id). name_to_id is keyed by lowercased
    exact DrugBank primary name -- see module docstring for the exact-match
    caveat (no synonym/brand-name table exists in this repo)."""

    id_to_name = {}
    name_to_id = {}

    for row in load_jsonl(DATA_DIR / "drugs.jsonl"):
        id_to_name[row["drugbank_id"]] = row["name"]
        name_to_id[row["name"].lower()] = row["drugbank_id"]

    return id_to_name, name_to_id


def resolve_drug(token, id_to_name, name_to_id):
    """token may be a DrugBank ID or an exact (case-insensitive) drug name."""

    if token in id_to_name:
        return token
    return name_to_id.get(token.lower())


def load_drug_smiles():
    """{drugbank_id: smiles}, from scripts/extract_smiles.py's output.
    Drugs without a SMILES (mostly biologics) simply aren't keys here --
    same absence-is-the-signal convention as load_per_drug_table's tables."""

    try:
        return {row["drugbank_id"]: row["smiles"] for row in load_jsonl(DATA_DIR / "drug_smiles.jsonl")}
    except FileNotFoundError:
        return {}


def load_mechanism_model(checkpoint_path, vocab_path, hidden_dims):
    """Lazy torch/model import -- only paid when --predict is actually used."""

    import torch
    from models.mechanism_mlp import MechanismMLP

    with open(vocab_path) as f:
        vocab_data = json.load(f)
    label_vocab = vocab_data["labels"]

    model = MechanismMLP(in_dim=1024, hidden_dims=tuple(hidden_dims), out_dim=len(label_vocab))
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
    model.eval()

    return model, label_vocab


def load_knn_pool(fingerprints_path, dataset_path, split_path):
    """Loads the k-NN ensemble's neighbor pool: the main train split's
    fingerprints + label vectors ONLY (never val/test -- the same
    population the linear checkpoint itself was trained on, so val/test
    stay a permanently honest held-out measurement). Returns
    (x_train [N, 1024], y_train [N, C]) tensors -- see
    scripts/mechanism_knn_transfer.py for how they're used."""

    import torch

    from scripts.train_mechanism_predictor import build_xy

    dataset = torch.load(dataset_path, weights_only=False)
    drug_ids, label_matrix = dataset["drug_ids"], dataset["labels"]
    label_by_drug = dict(zip(drug_ids, label_matrix))

    fp_data = torch.load(fingerprints_path, weights_only=False)
    fp_by_drug = dict(zip(fp_data["drug_ids"], fp_data["fingerprints"]))

    with open(split_path) as f:
        split = json.load(f)

    return build_xy(drug_ids, fp_by_drug, label_by_drug, split["train"])


def load_label_thresholds(path, label_vocab, flat_override=None):
    """Returns a list[float] of per-label thresholds, aligned to label_vocab.

    If flat_override is given, every label uses that single value --
    bypasses the tuned thresholds file entirely. Otherwise loads `path`
    (produced by scripts/tune_mechanism_thresholds.py: per-label F1-optimal
    where the val split has enough support, else a per-kind pooled fallback
    -- see scripts/mechanism_thresholds.py for why naive per-label tuning
    doesn't work on this small a val split). Falls back to a flat 0.5, with
    a printed warning, if the file is missing or its vocab doesn't match."""

    if flat_override is not None:
        return [flat_override] * len(label_vocab)

    try:
        with open(path) as f:
            data = json.load(f)
        if data["label_vocab"] != label_vocab:
            print(
                f"WARNING: {path}'s label vocab doesn't match {DEFAULT_PREDICT_VOCAB} -- "
                f"falling back to a flat 0.5 threshold for every label.",
                file=sys.stderr,
            )
            return [0.5] * len(label_vocab)
        return data["thresholds"]
    except FileNotFoundError:
        print(
            f"WARNING: {path} not found -- falling back to a flat 0.5 threshold "
            f"for every label. Run scripts/tune_mechanism_thresholds.py to generate it.",
            file=sys.stderr,
        )
        return [0.5] * len(label_vocab)


def predict_mechanisms(smiles, model, label_vocab, thresholds, knn_pool=None, knn_k=DEFAULT_KNN_K):
    """Returns {"enzyme": [...], "target": [...], "transporter": [...]} in
    the same {"name":, "actions": [...]} shape documented profiles use, plus
    a parallel "confidences" list aligned with "actions". Returns None if
    the SMILES doesn't parse -- caller should report
    "no_structural_fingerprint" rather than an empty (and misleadingly
    confident-looking) result.

    thresholds: list[float] aligned to label_vocab (see load_label_thresholds) --
    per-label, not one flat cutoff for every label. Must be tuned against
    the SAME probability source (linear-only vs. blend) passed here, or
    confidence cutoffs will be miscalibrated.

    knn_pool: optional (x_train, y_train) tensors from load_knn_pool -- if
    given, the linear model's probabilities are averaged with a Tanimoto
    k-NN vote over the pool (scripts/mechanism_knn_transfer.py) before
    thresholding. See the DEFAULT_KNN_K module comment for why this is the
    --predict default now.

    Fingerprint settings (radius/nBits/chirality) MUST match
    scripts/build_fingerprints.py, which is what the checkpoint was trained
    on -- see the DEFAULT_PREDICT_FP_* module constants.
    """

    import numpy as np
    import torch
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.error")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=DEFAULT_PREDICT_FP_RADIUS,
        fpSize=DEFAULT_PREDICT_FP_NBITS,
        includeChirality=DEFAULT_PREDICT_FP_CHIRALITY,
    )
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((DEFAULT_PREDICT_FP_NBITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)

    x = torch.from_numpy(arr).unsqueeze(0)
    with torch.no_grad():
        probs = torch.sigmoid(model(x)).squeeze(0)

    if knn_pool is not None:
        from scripts.mechanism_knn_transfer import knn_transfer_probs

        x_train, y_train = knn_pool
        knn_probs = knn_transfer_probs(x, x_train, y_train, k=knn_k).squeeze(0)
        probs = (probs + knn_probs) / 2

    grouped = {"enzyme": {}, "target": {}, "transporter": {}}

    for idx, label in enumerate(label_vocab):
        p = probs[idx].item()
        if p < thresholds[idx]:
            continue
        kind, protein, action = label.split(":", 2)
        entry = grouped[kind].setdefault(protein, {"actions": [], "confidences": []})
        entry["actions"].append(action)
        entry["confidences"].append(round(p, 3))

    return {
        kind: [{"name": name, **data} for name, data in proteins.items()]
        for kind, proteins in grouped.items()
    }


def merge_profiles_with_source(documented_entries, predicted_entries):
    """Combines a documented per-drug/per-kind profile list with a predicted
    one (see predict_mechanisms) into one list, tagging each entry's
    provenance. If the same protein appears in both, actions are unioned and
    the entry stays source='documented' (ground truth) with the model's
    confidence attached and also_predicted=True -- corroboration signal,
    not a conflict to silently resolve by dropping one side."""

    combined = {}

    for e in documented_entries:
        combined[e["name"]] = {
            "name": e["name"],
            "actions": list(e.get("actions", [])),
            "source": "documented",
        }

    for e in predicted_entries:
        confidence = max(e["confidences"]) if e.get("confidences") else None
        if e["name"] in combined:
            existing = combined[e["name"]]
            existing["actions"] = sorted(set(existing["actions"]) | set(e.get("actions", [])))
            existing["also_predicted"] = True
            existing["confidence"] = confidence
        else:
            combined[e["name"]] = {
                "name": e["name"],
                "actions": list(e.get("actions", [])),
                "source": "predicted",
                "confidence": confidence,
            }

    return list(combined.values())


def load_per_drug_table(path, key_field, id_field, drug_ids):
    """Load a {<id_field>, <key_field>, actions} table, filtered to
    drug_ids, grouped as {drug_id: [{"name": ..., "actions": [...]}, ...]}."""

    out = {d: [] for d in drug_ids}

    for row in load_jsonl(path):
        drug = row[id_field]
        if drug not in out:
            continue
        entry = {"name": row[key_field]}
        if "actions" in row:
            entry["actions"] = row["actions"]
        out[drug].append(entry)

    return out


def find_documented_interactions(drug_ids):
    """Stream drugbank_interactions.jsonl once, keeping only rows where BOTH
    drugs are in drug_ids. A cheap quoted-substring pre-check (bytes,
    precomputed once) avoids paying JSON-parse cost for the ~2.9M rows that
    can't possibly match.

    DrugBank lists most pairs twice, once per direction (drug1/drug2
    swapped); empirically these carry identical description text (verified:
    0 divergences across 695k+ directional duplicates in this dataset). We
    still collect ALL distinct descriptions per pair rather than blindly
    keeping whichever direction is seen last, so a future data refresh with
    genuinely different per-direction text wouldn't silently lose one side.
    """

    quoted_ids = {f'"{d}"'.encode() for d in drug_ids}
    found = {}  # frozenset({a, b}) -> set of distinct descriptions

    path = DATA_DIR / "drugbank_interactions.jsonl"
    with open(path, "rb") as f:
        for line in f:
            if not any(q in line for q in quoted_ids):
                continue
            row = orjson.loads(line)
            d1, d2 = row["drug1"], row["drug2"]
            if d1 in drug_ids and d2 in drug_ids:
                found.setdefault(frozenset((d1, d2)), set()).add(row["description"])

    return {key: " | ".join(sorted(descs)) for key, descs in found.items()}


def _format_profile_bullet(drug_id, name, profile):
    parts = []
    for label, entries in (
        ("enzymes", profile["enzymes"]),
        ("targets", profile["targets"]),
        ("transporters", profile["transporters"]),
        ("carriers", profile["carriers"]),
    ):
        if not entries:
            continue
        items = [
            f"{e['name']} ({', '.join(e['actions'])})" if e.get("actions") else e["name"]
            for e in entries
        ]
        parts.append(f"{label}: " + "; ".join(items))
    body = " | ".join(parts) if parts else "no enzyme/target/transporter/carrier data documented"
    return f"- {name} ({drug_id}): {body}"


def _format_predicted_profile_bullet(drug_id, name, profile):
    parts = []
    for label, key in (
        ("enzymes", "predicted_enzymes"),
        ("targets", "predicted_targets"),
        ("transporters", "predicted_transporters"),
    ):
        entries = profile.get(key, [])
        if not entries:
            continue
        items = [
            f"{e['name']} [" + ", ".join(
                f"{a} ({c:.2f})" for a, c in zip(e["actions"], e.get("confidences", []))
            ) + "]"
            for e in entries
        ]
        parts.append(f"{label}: " + "; ".join(items))
    if not parts:
        return None
    return f"- {name} ({drug_id}): " + " | ".join(parts)


def build_llm_prompt(report):
    """Builds part 2: a prompt that hands the part-1 mechanism findings to a
    downstream model, alongside placeholders for patient data and RAG-
    retrieved medical literature, so it can tell a pharmacist what will
    actually happen for THIS patient. This function only builds the prompt
    text -- filling in {{PATIENT_DATA}} / {{RETRIEVED_CONTEXT}} and calling
    the model is a separate downstream system's job, out of scope here.
    """

    drug_lines = "\n".join(
        f"- {d['name']} ({d['drugbank_id']})" for d in report["input_drugs"]
    )

    if report["unresolved_inputs"]:
        unknown_line = (
            "Unrecognized inputs (not a known DrugBank ID or exact drug "
            "name, excluded from the analysis below): "
            + ", ".join(report["unresolved_inputs"])
        )
    else:
        unknown_line = "All inputs were recognized."

    if report["documented_interactions"]:
        documented_lines = "\n".join(
            f"- {report['per_drug_profile'][x['drug_a']]['name']} + "
            f"{report['per_drug_profile'][x['drug_b']]['name']}: {x['description']}"
            for x in report["documented_interactions"]
        )
    else:
        documented_lines = "None documented by DrugBank for any pair in this combination."

    if report["mechanistic_overlaps"]:
        overlap_lines = []
        for x in report["mechanistic_overlaps"]:
            name_a = report["per_drug_profile"][x["drug_a"]]["name"]
            name_b = report["per_drug_profile"][x["drug_b"]]["name"]
            for kind, key in (
                ("enzyme", "shared_enzymes"),
                ("target", "shared_targets"),
                ("transporter", "shared_transporters"),
                ("carrier", "shared_carriers"),
            ):
                for e in x[key]:
                    line = (
                        f"- {name_a} + {name_b} both act on {kind} '{e['name']}' "
                        f"({name_a}: {', '.join(e['actions_a']) or 'unspecified action'}; "
                        f"{name_b}: {', '.join(e['actions_b']) or 'unspecified action'})"
                    )
                    for pk_effect in e.get("likely_pk_effects", []):
                        line += f"\n  -> rule-based PK reading: {pk_effect}"
                    overlap_lines.append(line)
        overlap_lines = "\n".join(overlap_lines)
    else:
        overlap_lines = "None found -- no shared enzymes, targets, transporters, or carriers between any pair."

    profile_lines = "\n".join(
        _format_profile_bullet(d["drugbank_id"], d["name"], report["per_drug_profile"][d["drugbank_id"]])
        for d in report["input_drugs"]
    )

    predicted_section = ""
    if "predicted_mechanistic_overlaps" in report:

        pred_overlap_lines = []
        for x in report["predicted_mechanistic_overlaps"]:
            name_a = report["per_drug_profile"][x["drug_a"]]["name"]
            name_b = report["per_drug_profile"][x["drug_b"]]["name"]
            for kind, key in (
                ("enzyme", "shared_enzymes"),
                ("target", "shared_targets"),
                ("transporter", "shared_transporters"),
                ("carrier", "shared_carriers"),
            ):
                for e in x[key]:
                    # Skip entries already shown in the documented-only
                    # section above -- this section is for NEW hints only.
                    if e.get("source_a") != "predicted" and e.get("source_b") != "predicted":
                        continue
                    tag_a = e.get("source_a", "documented")
                    if "confidence_a" in e:
                        tag_a += f" {e['confidence_a']:.2f}"
                    tag_b = e.get("source_b", "documented")
                    if "confidence_b" in e:
                        tag_b += f" {e['confidence_b']:.2f}"
                    line = (
                        f"- {name_a} + {name_b} both act on {kind} '{e['name']}' "
                        f"({name_a}: {', '.join(e['actions_a']) or 'unspecified action'} [{tag_a}]; "
                        f"{name_b}: {', '.join(e['actions_b']) or 'unspecified action'} [{tag_b}])"
                    )
                    for pk_effect in e.get("likely_pk_effects", []):
                        line += f"\n  -> rule-based PK reading: {pk_effect}"
                    pred_overlap_lines.append(line)

        pred_profile_lines = []
        for d in report["input_drugs"]:
            profile = report["per_drug_profile"][d["drugbank_id"]]
            if profile.get("structural_prediction") == "no_structural_fingerprint":
                pred_profile_lines.append(
                    f"- {d['name']} ({d['drugbank_id']}): no structural prediction available "
                    f"(no usable SMILES/fingerprint -- likely a biologic)."
                )
                continue
            bullet = _format_predicted_profile_bullet(d["drugbank_id"], d["name"], profile)
            if bullet:
                pred_profile_lines.append(bullet)

        pred_overlap_text = "\n".join(pred_overlap_lines) if pred_overlap_lines else (
            "None -- no NEW shared enzyme/target/transporter beyond what's already "
            "documented above was predicted for any pair."
        )
        pred_profile_text = "\n".join(pred_profile_lines) if pred_profile_lines else (
            "No structure-based predictions available for any input drug."
        )

        if report.get("predict_method") == "blend":
            method_ap_clause = (
                "a MODEL GUESS from each drug's chemical structure alone, blending a linear "
                "model with a Tanimoto-similarity k-NN vote over structurally similar training "
                "drugs, 5-fold cross-validated to macro-AP 0.293 +/- 0.016 / micro-AP 0.184 +/- "
                "0.017 (mean +/- std across folds, beats the linear model alone in 5/5 folds -- "
                "see findings/findings.md's \"Item 2 result\")"
            )
        else:
            method_ap_clause = (
                "a MODEL GUESS from each drug's chemical structure alone, 5-fold cross-validated "
                "to macro-AP 0.258 +/- 0.013 / micro-AP 0.146 +/- 0.008 (mean +/- std across "
                "folds, not a single lucky split -- see findings/findings.md)"
            )

        predicted_section = f"""## Predicted mechanisms (structure-based ML, NOT DrugBank-documented -- {method_ap_clause} \
-- treat as a weak hint, not a fact, and weigh DOCUMENTED/GROUNDED evidence above it first)

New shared-mechanism hints beyond the documented section above:
{pred_overlap_text}

Per-drug predicted profile:
{pred_profile_text}

"""

    return f"""You are a clinical pharmacology assistant supporting a pharmacist in \
evaluating a polypharmacy regimen. You are given three things: (1) a \
deterministic, DrugBank-sourced mechanism report for a specific drug \
combination, (2) this patient's clinical data, and (3) excerpts retrieved \
from medical literature and drug monographs relevant to this case. Use all \
three together -- do not rely on the mechanism report alone, and do not rely \
on general knowledge where the retrieved literature or patient data should \
govern instead.

## Drug combination
{drug_lines}

{unknown_line}

## DrugBank-documented interactions
{documented_lines}

## Mechanistic overlaps (NOT documented DrugBank interactions -- inferred by \
this pipeline from shared enzyme/target/transporter/carrier action; may or \
may not be clinically significant, and may or may not already be covered by \
a documented interaction above)
{overlap_lines}

{predicted_section}## Per-drug pharmacology profile (DrugBank)
{profile_lines}

## Patient data
{{{{PATIENT_DATA}}}}

## Retrieved medical literature (RAG)
{{{{RETRIEVED_CONTEXT}}}}

## Task
Using the mechanism report above together with this patient's data and the \
retrieved literature, explain to the pharmacist:
1. What clinically relevant effect(s) this drug combination is likely to \
have in THIS patient specifically (not just in general).
2. Whether each documented interaction and mechanistic overlap above is \
actually relevant given this patient's data (e.g. renal/hepatic function, \
other conditions, other medications).
3. Recommended monitoring, dose adjustment, or alternative therapy \
considerations, citing the retrieved literature where possible.
4. Your confidence, explicitly distinguishing what is DOCUMENTED (DrugBank \
interaction text), what is INFERRED (mechanistic overlap only), and what is \
GROUNDED in the retrieved literature or patient data.

Do not state a claim as established fact unless it is supported by the \
mechanism report, the patient data, or the retrieved literature above. If \
evidence conflicts or is insufficient to answer confidently, say so \
explicitly rather than guessing."""


def infer_pk_effects(actions_a, actions_b, name_a, name_b):
    """Deterministic, rule-based restatement of what pairing these two
    DrugBank action sets on a SHARED enzyme/transporter conventionally means
    pharmacokinetically (inhibitor+substrate -> raised exposure,
    inducer+substrate -> lowered exposure/efficacy, substrate+substrate ->
    competition). This is textbook PK terminology, not a model prediction --
    but it's still a rule-based INTERPRETATION layered on top of the raw
    DrugBank facts, so callers should treat it as a hint for the downstream
    reasoning step, not a clinical conclusion on its own (adapted from the
    equivalent, previously-unused logic in scripts/mechanism_engine.py).
    """

    a, b = set(actions_a), set(actions_b)
    effects = []

    if "inhibitor" in a and "substrate" in b:
        effects.append(f"{name_a} may increase exposure to {name_b}")
    if "inhibitor" in b and "substrate" in a:
        effects.append(f"{name_b} may increase exposure to {name_a}")
    if "inducer" in a and "substrate" in b:
        effects.append(f"{name_a} may reduce exposure/efficacy of {name_b}")
    if "inducer" in b and "substrate" in a:
        effects.append(f"{name_b} may reduce exposure/efficacy of {name_a}")
    if "substrate" in a and "substrate" in b:
        effects.append(f"potential competitive metabolism/transport between {name_a} and {name_b}")

    return effects


def shared(profile_a, profile_b, name_a=None, name_b=None, infer_pk=False, carry_source=False):
    """carry_source=False (default, used by every pre-existing call site)
    keeps this function's output byte-for-byte identical to before --predict
    existed. Pass carry_source=True (only done for the new --predict path,
    with merge_profiles_with_source()-built inputs) to additionally surface
    which side of a shared protein was documented vs. model-predicted."""

    names_a = {e["name"] for e in profile_a}
    names_b = {e["name"] for e in profile_b}
    overlap = names_a & names_b
    if not overlap:
        return []
    by_name_a = {e["name"]: e for e in profile_a}
    by_name_b = {e["name"]: e for e in profile_b}

    result = []
    for name in sorted(overlap):
        ea, eb = by_name_a[name], by_name_b[name]
        actions_a, actions_b = ea.get("actions", []), eb.get("actions", [])
        entry = {"name": name, "actions_a": actions_a, "actions_b": actions_b}
        if carry_source:
            entry["source_a"] = ea.get("source", "documented")
            entry["source_b"] = eb.get("source", "documented")
            if ea.get("confidence") is not None:
                entry["confidence_a"] = ea["confidence"]
            if eb.get("confidence") is not None:
                entry["confidence_b"] = eb["confidence"]
        if infer_pk:
            entry["likely_pk_effects"] = infer_pk_effects(actions_a, actions_b, name_a, name_b)
        result.append(entry)

    return result


def main():
    args = parse_args()
    requested = list(dict.fromkeys(args.drug_ids))  # de-dupe, preserve order

    if len(requested) < 2:
        raise SystemExit("Need at least 2 DrugBank IDs and/or drug names.")

    names, name_to_id = load_drug_index()

    resolved = [resolve_drug(token, names, name_to_id) for token in requested]
    valid = list(dict.fromkeys(d for d in resolved if d is not None))  # de-dupe post-resolution
    unresolved = [tok for tok, d in zip(requested, resolved) if d is None]

    if len(valid) < 2:
        raise SystemExit(
            f"Need at least 2 KNOWN drugs to compare; only resolved "
            f"{valid} against data/processed/drugs.jsonl. Unresolved: {unresolved}"
        )

    print(f"Loading mechanism tables for {len(valid)} drugs...", file=sys.stderr)

    enzymes = load_per_drug_table(DATA_DIR / "drug_enzymes.jsonl", "enzyme", "drug", valid)
    targets = load_per_drug_table(DATA_DIR / "drug_targets.jsonl", "target", "drug", valid)
    transporters = load_per_drug_table(DATA_DIR / "drug_transporters.jsonl", "transporter", "drug", valid)
    carriers = load_per_drug_table(DATA_DIR / "drug_carriers.jsonl", "carrier", "drug", valid)

    print("Scanning documented DrugBank interactions...", file=sys.stderr)
    documented = find_documented_interactions(set(valid))

    per_drug_profile = {
        d: {
            "name": names[d],
            "enzymes": enzymes[d],
            "targets": targets[d],
            "transporters": transporters[d],
            "carriers": carriers[d],
        }
        for d in valid
    }

    predicted_by_drug = {}  # drug_id -> {"enzyme": [...], "target": [...], "transporter": [...]} or None

    if args.predict:

        print("Loading mechanism gap-filler model...", file=sys.stderr)
        model, label_vocab = load_mechanism_model(
            args.predict_checkpoint, args.predict_vocab, args.predict_hidden_dims
        )

        knn_pool = None
        if not args.predict_no_knn_blend:
            print("Loading k-NN ensemble neighbor pool...", file=sys.stderr)
            knn_pool = load_knn_pool(
                args.predict_knn_fingerprints, args.predict_knn_dataset, args.predict_knn_split
            )

        thresholds_path = args.predict_thresholds_path or (
            DEFAULT_PREDICT_THRESHOLDS if args.predict_no_knn_blend else DEFAULT_PREDICT_THRESHOLDS_BLEND
        )
        thresholds = load_label_thresholds(
            thresholds_path, label_vocab, flat_override=args.predict_threshold
        )
        smiles_by_drug = load_drug_smiles()

        for d in valid:

            smiles = smiles_by_drug.get(d)

            predicted = (
                predict_mechanisms(
                    smiles, model, label_vocab, thresholds,
                    knn_pool=knn_pool, knn_k=args.predict_knn_k,
                )
                if smiles else None
            )
            predicted_by_drug[d] = predicted

            if predicted is None:
                per_drug_profile[d]["structural_prediction"] = "no_structural_fingerprint"
                per_drug_profile[d]["predicted_enzymes"] = []
                per_drug_profile[d]["predicted_targets"] = []
                per_drug_profile[d]["predicted_transporters"] = []
            else:
                per_drug_profile[d]["structural_prediction"] = "ok"
                per_drug_profile[d]["predicted_enzymes"] = predicted["enzyme"]
                per_drug_profile[d]["predicted_targets"] = predicted["target"]
                per_drug_profile[d]["predicted_transporters"] = predicted["transporter"]

    documented_interactions = []
    mechanistic_overlaps = []
    predicted_mechanistic_overlaps = []

    for a, b in combinations(valid, 2):
        pair_key = frozenset((a, b))

        if pair_key in documented:
            documented_interactions.append(
                {"drug_a": a, "drug_b": b, "description": documented[pair_key]}
            )

        enzyme_overlap = shared(enzymes[a], enzymes[b], names[a], names[b], infer_pk=True)
        target_overlap = shared(targets[a], targets[b])
        transporter_overlap = shared(transporters[a], transporters[b], names[a], names[b], infer_pk=True)
        carrier_overlap = shared(carriers[a], carriers[b])

        if enzyme_overlap or target_overlap or transporter_overlap or carrier_overlap:
            mechanistic_overlaps.append(
                {
                    "drug_a": a,
                    "drug_b": b,
                    "shared_enzymes": enzyme_overlap,
                    "shared_targets": target_overlap,
                    "shared_transporters": transporter_overlap,
                    "shared_carriers": carrier_overlap,
                    "note": (
                        "Shared enzyme/transporter/target/carrier action, "
                        "computed by intersecting each drug's individual "
                        "DrugBank profile. This is NOT a documented DrugBank "
                        "interaction -- it's a plausible PK/PD mechanism (e.g. "
                        "both drugs inhibit the same CYP enzyme) that may or "
                        "may not be clinically significant, and may or may "
                        "not already be covered by a documented_interactions "
                        "entry above."
                    ),
                }
            )

        if args.predict:

            pred_a, pred_b = predicted_by_drug[a] or {}, predicted_by_drug[b] or {}

            combined_enzymes_a = merge_profiles_with_source(enzymes[a], pred_a.get("enzyme", []))
            combined_enzymes_b = merge_profiles_with_source(enzymes[b], pred_b.get("enzyme", []))
            combined_targets_a = merge_profiles_with_source(targets[a], pred_a.get("target", []))
            combined_targets_b = merge_profiles_with_source(targets[b], pred_b.get("target", []))
            combined_transporters_a = merge_profiles_with_source(transporters[a], pred_a.get("transporter", []))
            combined_transporters_b = merge_profiles_with_source(transporters[b], pred_b.get("transporter", []))
            combined_carriers_a = merge_profiles_with_source(carriers[a], [])
            combined_carriers_b = merge_profiles_with_source(carriers[b], [])

            pred_enzyme_overlap = shared(
                combined_enzymes_a, combined_enzymes_b, names[a], names[b],
                infer_pk=True, carry_source=True,
            )
            pred_target_overlap = shared(
                combined_targets_a, combined_targets_b, carry_source=True,
            )
            pred_transporter_overlap = shared(
                combined_transporters_a, combined_transporters_b, names[a], names[b],
                infer_pk=True, carry_source=True,
            )
            pred_carrier_overlap = shared(
                combined_carriers_a, combined_carriers_b, carry_source=True,
            )

            if (pred_enzyme_overlap or pred_target_overlap
                    or pred_transporter_overlap or pred_carrier_overlap):
                predicted_mechanistic_overlaps.append(
                    {
                        "drug_a": a,
                        "drug_b": b,
                        "shared_enzymes": pred_enzyme_overlap,
                        "shared_targets": pred_target_overlap,
                        "shared_transporters": pred_transporter_overlap,
                        "shared_carriers": pred_carrier_overlap,
                        "note": (
                            "Same as mechanistic_overlaps' note, but computed "
                            "over documented+predicted combined profiles -- "
                            "each entry's source_a/source_b says whether that "
                            "side came from DrugBank ('documented') or the "
                            "structure-based ML gap-filler ('predicted'). "
                            "Superset of mechanistic_overlaps above; entries "
                            "with both sides 'documented' just duplicate what's "
                            "already there."
                        ),
                    }
                )

    report = {
        "source": "DrugBank (deterministic lookup, no ML/LLM)",
        "input_drugs": [{"drugbank_id": d, "name": names[d]} for d in valid],
        "unresolved_inputs": unresolved,
        "per_drug_profile": per_drug_profile,
        "documented_interactions": documented_interactions,
        "mechanistic_overlaps": mechanistic_overlaps,
    }

    if args.predict:
        report["predicted_mechanistic_overlaps"] = predicted_mechanistic_overlaps
        report["predict_method"] = "linear" if args.predict_no_knn_blend else "blend"

    output = {
        "part_1_mechanism_report": report,
        "part_2_downstream_prompt": build_llm_prompt(report),
    }

    text = json.dumps(output, indent=2 if args.pretty else None)

    if args.out:
        Path(args.out).write_text(text)
        print(f"Saved report to {args.out}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
