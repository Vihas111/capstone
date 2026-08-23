"""FastAPI backend for the DDI web UI, wired directly to the Capstone
project's real models -- NO LLM, NO RAG (explicitly out of scope for this
UI; see /Volumes/Elements/Capstone/CLAUDE.md).

Pipeline per request (POST /predict {drugs: [name_or_id, ...]}, 2 or 3 drugs):
    resolve drug names -> DrugBank per-drug profile lookup (deterministic)
    -> documented DrugBank interaction text (deterministic)
    -> mechanistic overlap (deterministic, shared enzyme/target/
       transporter/carrier) + structure-based ML gap-filler overlap
    -> two independently trained pairwise link classifiers:
         Model 1: spectral Hetionet embedding + HistGradientBoosting
                  (5-fold ROC-AUC 0.784 +/- 0.005)
         Model 2: task-supervised GCN over Hetionet
                  (5-fold ROC-AUC 0.945 +/- 0.006, transductive)
    -> a deterministically-templated summary (NOT LLM-written) + a fixed
       caveat, reusing the same response fields the old LLM/RAG-based API
       returned so the frontend needs no type changes.

For 3 drugs, every one of the C(3,2)=3 pairs is scored independently
(pairwise decomposition -- see ml.LINK_PREDICTION_CAVEAT, surfaced in the
response when more than one pair is scored) and the results are merged.

All actual computation is delegated to scripts/mechanism_lookup.py and
scripts/predict_multidrug_interaction.py in the main Capstone repo -- this
file is glue: load everything ONCE at startup, then call the same
functions those CLI tools call, per request, with the same defaults.

NOTE on load_model2: predict_multidrug_interaction.py's load_model2() calls
torch.load(decoder_path, weights_only=True) with no map_location, which
crashes on a CPU-only machine loading a checkpoint saved on a CUDA machine
(RuntimeError: "Attempting to deserialize object on a CUDA device..."). This
is a real portability bug (same family as the venv/checkpoint symlink
issues already documented in that project's CLAUDE.md) -- worth fixing
upstream, but the Capstone repo's external drive is mounted read-only on
this machine (`mount`: `ntfs ... read-only`), so it can't be patched in
place from here. Reimplemented locally below with an explicit
map_location="cpu" instead.
"""

import logging
import os
import sys
import time
from itertools import combinations
from pathlib import Path

from threadpoolctl import threadpool_limits

WEBAPI_ROOT = str(Path(__file__).resolve().parent)
sys.path.insert(0, WEBAPI_ROOT)  # for `import convergence` (this dir), unaffected by the chdir below

CAPSTONE_ROOT = "/Volumes/Elements/Capstone"
os.chdir(CAPSTONE_ROOT)  # DATA_DIR / drugs.jsonl etc. in mechanism_lookup.py are cwd-relative
sys.path.insert(0, CAPSTONE_ROOT)

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scripts import mechanism_lookup as ml
from scripts import predict_multidrug_interaction as pmi
from scripts.run_pair_link_classifier_gnn_kfold import PairDecoder

# All local to webapi/, none part of the read-only Capstone repo.
import convergence
import outcomes
from knowledge import KB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ddi.webapi")

CORS_ORIGINS = [
    o.strip()
    for o in os.environ.get(
        "CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    if o.strip()
]

MIN_DRUGS = 2
MAX_DRUGS = 3  # the web UI's toggle only offers 2 or 3; the underlying scripts support up to 4

app = FastAPI(title="DDI Model API (no LLM/RAG)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state: dict = {}


def _load_model2_cpu(embeddings_path, decoder_path):
    """Same as pmi.load_model2, but safe on a CPU-only machine. See module
    docstring: pmi.load_model2 doesn't pass map_location and crashes here."""
    data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    node_index, embeddings, out_dim = data["node_index"], data["embeddings"], data["out_dim"]
    decoder = PairDecoder(out_dim)
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
    decoder.eval()
    return node_index, embeddings, decoder


@app.on_event("startup")
def _startup():
    t0 = time.time()
    log.info("Loading drug index...")
    id_to_name, name_to_id = ml.load_drug_index()

    log.info("Loading reference knowledge (profiles, categories, risk axes)...")
    KB.load(log=log)

    log.info("Loading link-prediction Model 1 (spectral + HGB)...")
    try:
        clf, cache = pmi.load_model1(pmi.DEFAULT_HGB_MODEL, pmi.DEFAULT_HGB_CACHE)
    except FileNotFoundError as e:
        log.warning("Model 1 unavailable: %s", e)
        clf, cache = None, None

    log.info("Loading link-prediction Model 2 (task-supervised GCN)...")
    try:
        node_index, embeddings, decoder = _load_model2_cpu(
            pmi.DEFAULT_GNN_EMBEDDINGS, pmi.DEFAULT_GNN_DECODER
        )
    except FileNotFoundError as e:
        log.warning("Model 2 unavailable: %s", e)
        node_index, embeddings, decoder = None, None, None

    log.info("Loading mechanism gap-filler model + k-NN blend pool...")
    try:
        mech_model, label_vocab = ml.load_mechanism_model(
            ml.DEFAULT_PREDICT_CHECKPOINT, ml.DEFAULT_PREDICT_VOCAB, ml.DEFAULT_PREDICT_HIDDEN_DIMS
        )
        knn_pool = ml.load_knn_pool(
            ml.DEFAULT_KNN_FINGERPRINTS, ml.DEFAULT_KNN_DATASET, ml.DEFAULT_KNN_SPLIT
        )
        thresholds = ml.load_label_thresholds(ml.DEFAULT_PREDICT_THRESHOLDS_BLEND, label_vocab)
        smiles_by_drug = ml.load_drug_smiles()
    except FileNotFoundError as e:
        log.warning("Mechanism gap-filler unavailable: %s", e)
        mech_model, label_vocab, knn_pool, thresholds, smiles_by_drug = None, None, None, None, {}

    _state.update(
        id_to_name=id_to_name,
        name_to_id=name_to_id,
        clf=clf,
        cache=cache,
        node_index=node_index,
        embeddings=embeddings,
        decoder=decoder,
        mech_model=mech_model,
        label_vocab=label_vocab,
        knn_pool=knn_pool,
        thresholds=thresholds,
        smiles_by_drug=smiles_by_drug,
    )
    log.info("Startup complete in %.1fs. %d drugs indexed.", time.time() - t0, len(id_to_name))


class PredictRequest(BaseModel):
    drugs: list[str]


# Interaction STATUS, not severity. Deliberately not a HIGH/MODERATE/LOW risk
# grade: nothing in this pipeline was ever trained or validated against
# severity-labelled data. DrugBank's interaction rows (as extracted here)
# carry no clinical severity grade, and both link classifiers were trained on
# a single binary question -- "does a documented interaction exist for this
# pair" -- not on how dangerous it is. An earlier version of this file graded
# any documented interaction as "HIGH", which put a trivial PK footnote
# (acetaminophen slightly slowing amoxicillin excretion) in the same visual
# bucket as warfarin+aspirin bleeding risk. These three states are claims
# about EVIDENCE PROVENANCE, which is what the data actually supports.
STATUS_RANK = {"none_found": 0, "predicted": 1, "documented": 2}

PREDICTED_THRESHOLD = 0.5


def _interaction_status(doc_text, score1, score2):
    if doc_text:
        return "documented"
    scores = [s for s in (score1, score2) if s is not None]
    if scores and max(scores) >= PREDICTED_THRESHOLD:
        return "predicted"
    return "none_found"


def _overlap_entries(kind, entries):
    out = []
    for e in entries:
        out.append(
            {
                "kind": kind,
                "protein": e["name"],
                "actions_a": e.get("actions_a", []),
                "actions_b": e.get("actions_b", []),
                "likely_pk_effects": e.get("likely_pk_effects", []),
                "source_a": e.get("source_a", "documented"),
                "source_b": e.get("source_b", "documented"),
            }
        )
    return out


def _build_grounding(name_a, name_b, doc_text, overlaps, predicted_overlaps):
    grounding = []
    if doc_text:
        grounding.append(
            {
                "drug": f"{name_a} + {name_b}",
                "section": "documented interaction",
                "text": doc_text,
            }
        )
    for o in overlaps:
        pk = " ".join(o["likely_pk_effects"]) if o["likely_pk_effects"] else ""
        text = (
            f"Both drugs act on {o['protein']}: {name_a} ({', '.join(o['actions_a']) or 'no action listed'}), "
            f"{name_b} ({', '.join(o['actions_b']) or 'no action listed'}). "
            f"{pk} Not a documented interaction -- a plausible shared PK/PD mechanism inferred by intersecting "
            f"each drug's individual DrugBank profile."
        ).strip()
        grounding.append(
            {"drug": f"{name_a} & {name_b}", "section": f"shared {o['kind']}: {o['protein']}", "text": text}
        )
    for o in predicted_overlaps:
        text = (
            f"{o['source_a']}/{o['source_b']} overlap on {o['protein']} -- at least one side is a structure-based "
            f"ML gap-filler guess (k-NN/linear blend, 5-fold macro-AP 0.293 +/- 0.016), not a confirmed DrugBank fact."
        )
        grounding.append(
            {
                "drug": f"{name_a} & {name_b}",
                "section": f"predicted shared {o['kind']}: {o['protein']}",
                "text": text,
            }
        )
    return grounding


def _build_explanation(name_a, name_b, doc_text, overlaps, score1, score2):
    """A single, plain-language sentence stating what the actual clinical
    concern IS -- distinct from the Evidence accordion (raw facts, one per
    item, requires reading several) and the risk-model bars (a number with
    no explanation of what it means). Priority: a real DrugBank-documented
    interaction is the most authoritative and direct answer to "what's the
    problem" when one exists; failing that, the most concrete inferred
    PK/PD effect from a shared mechanism; failing that, name the shared
    mechanism itself; only if there's truly nothing do we fall back to a
    plain "nothing documented" statement (in which case the risk-model
    scores below are the only signal, and the sentence says so)."""
    if doc_text:
        return doc_text

    for o in overlaps:
        if o["likely_pk_effects"]:
            return (
                f"{o['likely_pk_effects'][0]} -- inferred from a shared {o['kind']} "
                f"({o['protein']}), not a documented DrugBank interaction."
            )

    if overlaps:
        proteins = ", ".join(sorted({o["protein"] for o in overlaps})[:3])
        return (
            f"No documented DrugBank interaction, but {name_a} and {name_b} both act on {proteins} "
            f"-- a plausible but unconfirmed shared mechanism."
        )

    scores = [s for s in (score1, score2) if s is not None]
    if scores and max(scores) >= 0.5:
        return (
            f"No documented interaction and no shared enzyme/target/transporter/carrier mechanism found "
            f"between {name_a} and {name_b} -- this pair is flagged solely on structural/graph similarity "
            f"to other known interacting drugs, not on any specific known mechanism."
        )

    return f"No documented interaction or shared mechanism found between {name_a} and {name_b}."


NOTE_BASE = (
    "Deterministic DrugBank lookups plus two ML classifiers -- no LLM. Decision support, not clinical "
    "advice; verify with a pharmacist. Note: this reports whether an interaction exists, not how severe "
    "it is -- nothing here is validated against severity-graded data."
)
NOTE_MULTI_PAIR = (
    "Scores are pairwise: a 3-drug combination could interact in ways no single pair shows."
)
NOTE_CONVERGENCE = (
    "Convergence findings are rule-based readings of DrugBank action verbs, not clinical conclusions."
)


def _score_pair(a: str, b: str, documented: dict, enzymes: dict, targets: dict, transporters: dict, carriers: dict) -> dict:
    """All computation for one drug pair. Returns everything needed to
    build that pair's slice of the final response, plus per-stage timings.

    `documented` is the {frozenset(pair): text} map for ALL pairs in this
    request, and `enzymes`/`targets`/`transporters`/`carriers` are the
    {drug_id: [...]} profile tables for ALL valid drugs in this request --
    both computed ONCE by the caller and shared across every pair, not
    reloaded per pair. find_documented_interactions() does a full linear
    scan of a 463MB file per call, and load_per_drug_table() a full scan of
    its own file; both accept any number of drug ids at once, so calling
    them once per pair (as an earlier version of this function did) was
    doing those scans N times over for an N-pair request instead of once.
    On this machine those files live on a slow, read-only, external
    NTFS-over-USB drive, so a cold read is expensive -- redundant scans
    compound badly. Loading every valid drug's profile once up front (not
    just each pair's two) also lets the caller run N-way convergence
    detection (convergence.py) across the whole combination, not just
    pairs -- see that module's docstring for why that matters."""
    id_to_name = _state["id_to_name"]
    name_a, name_b = id_to_name[a], id_to_name[b]
    timings: dict = {}

    t1 = time.time()
    doc_text = documented.get(frozenset((a, b)))
    timings["lookup"] = round((time.time() - t1) * 1000, 1)

    t2 = time.time()
    overlaps = (
        _overlap_entries("enzyme", ml.shared(enzymes[a], enzymes[b], name_a, name_b, infer_pk=True))
        + _overlap_entries("target", ml.shared(targets[a], targets[b]))
        + _overlap_entries("transporter", ml.shared(transporters[a], transporters[b], name_a, name_b, infer_pk=True))
        + _overlap_entries("carrier", ml.shared(carriers[a], carriers[b]))
    )
    timings["mechanistic_overlap"] = round((time.time() - t2) * 1000, 1)

    predicted_overlaps = []
    if _state["mech_model"] is not None:
        t3 = time.time()
        smiles_by_drug = _state["smiles_by_drug"]
        predicted_by_drug = {}
        for d in (a, b):
            smiles = smiles_by_drug.get(d)
            predicted_by_drug[d] = (
                ml.predict_mechanisms(
                    smiles, _state["mech_model"], _state["label_vocab"], _state["thresholds"],
                    knn_pool=_state["knn_pool"],
                )
                if smiles else None
            )
        pred_a, pred_b = predicted_by_drug[a] or {}, predicted_by_drug[b] or {}
        combined_enzymes_a = ml.merge_profiles_with_source(enzymes[a], pred_a.get("enzyme", []))
        combined_enzymes_b = ml.merge_profiles_with_source(enzymes[b], pred_b.get("enzyme", []))
        combined_targets_a = ml.merge_profiles_with_source(targets[a], pred_a.get("target", []))
        combined_targets_b = ml.merge_profiles_with_source(targets[b], pred_b.get("target", []))
        combined_transporters_a = ml.merge_profiles_with_source(transporters[a], pred_a.get("transporter", []))
        combined_transporters_b = ml.merge_profiles_with_source(transporters[b], pred_b.get("transporter", []))
        predicted_overlaps = (
            _overlap_entries(
                "enzyme",
                ml.shared(combined_enzymes_a, combined_enzymes_b, name_a, name_b, infer_pk=True, carry_source=True),
            )
            + _overlap_entries("target", ml.shared(combined_targets_a, combined_targets_b, carry_source=True))
            + _overlap_entries(
                "transporter",
                ml.shared(
                    combined_transporters_a, combined_transporters_b, name_a, name_b,
                    infer_pk=True, carry_source=True,
                ),
            )
        )
        predicted_overlaps = [o for o in predicted_overlaps if o["source_a"] == "predicted" or o["source_b"] == "predicted"]
        timings["gap_filler"] = round((time.time() - t3) * 1000, 1)

    t4 = time.time()
    score1 = score2 = None
    if _state["clf"] is not None:
        # HistGradientBoostingClassifier.predict_proba spins up an OpenMP/BLAS
        # thread pool per call; for a single-row prediction that spin-up cost
        # (seconds, on this machine) dwarfs the actual tree traversal. Limiting
        # to 1 thread avoids the overhead entirely -- confirmed 11.1s -> 0.011s
        # for one pair, same process, same model.
        with threadpool_limits(limits=1):
            score1 = pmi.score_pair_model1(_state["clf"], _state["cache"], a, b)
    if _state["decoder"] is not None:
        score2 = pmi.score_pair_model2(_state["node_index"], _state["embeddings"], _state["decoder"], a, b)
    timings["link_models"] = round((time.time() - t4) * 1000, 1)

    grounding = _build_grounding(name_a, name_b, doc_text, overlaps, predicted_overlaps)
    explanation = _build_explanation(name_a, name_b, doc_text, overlaps, score1, score2)
    status = _interaction_status(doc_text, score1, score2)

    return {
        "name_a": name_a,
        "name_b": name_b,
        "score1": score1,
        "score2": score2,
        "status": status,
        "grounding": grounding,
        "explanation": explanation,
        "timings": timings,
    }


@app.post("/predict")
def predict_endpoint(req: PredictRequest):
    t0 = time.time()
    id_to_name, name_to_id = _state["id_to_name"], _state["name_to_id"]
    requested = list(dict.fromkeys(d.strip() for d in req.drugs if d.strip()))

    if not (MIN_DRUGS <= len(requested) <= MAX_DRUGS):
        return {
            "found": False,
            "error": f"This tool supports {MIN_DRUGS}-{MAX_DRUGS} drugs (got {len(requested)}).",
            "error_type": "unsupported_drug_count",
            "drugs": requested,
            "drug_ids": [],
            "predicted_ades": [],
            "grounding": [],
            "explanation": None,
            "report": None,
            "meta": {"request_id": None, "timings_ms": {}, "use_real_model": True},
        }

    resolved = [ml.resolve_drug(d, id_to_name, name_to_id) for d in requested]
    unresolved = [d for d, r in zip(requested, resolved) if r is None]
    valid = list(dict.fromkeys(r for r in resolved if r is not None))

    if len(valid) < 2:
        return {
            "found": False,
            "error": f"Could not resolve: {', '.join(unresolved)}" if unresolved else "Need at least 2 known drugs.",
            "error_type": "drug_not_found",
            "drugs": requested,
            "drug_ids": resolved,
            "predicted_ades": [],
            "grounding": [],
            "explanation": None,
            "report": None,
            "meta": {"request_id": None, "timings_ms": {}, "use_real_model": True},
        }

    pairs = list(combinations(valid, 2))
    multi_pair = len(pairs) > 1

    t_doc = time.time()
    documented = ml.find_documented_interactions(set(valid))  # one scan for the whole request, see _score_pair
    doc_scan_ms = round((time.time() - t_doc) * 1000, 1)

    t_profiles = time.time()
    enzymes = ml.load_per_drug_table(ml.DATA_DIR / "drug_enzymes.jsonl", "enzyme", "drug", valid)
    targets = ml.load_per_drug_table(ml.DATA_DIR / "drug_targets.jsonl", "target", "drug", valid)
    transporters = ml.load_per_drug_table(ml.DATA_DIR / "drug_transporters.jsonl", "transporter", "drug", valid)
    carriers = ml.load_per_drug_table(ml.DATA_DIR / "drug_carriers.jsonl", "carrier", "drug", valid)
    profile_scan_ms = round((time.time() - t_profiles) * 1000, 1)

    pair_results = [_score_pair(a, b, documented, enzymes, targets, transporters, carriers) for a, b in pairs]

    # --- multi-channel detection (channels A, B, D) ---
    analysis = convergence.analyse(valid, id_to_name)

    # --- channel C: outcomes DrugBank itself names in its interaction text ---
    documented_texts = [
        (id_to_name[a], id_to_name[b], documented[frozenset((a, b))])
        for a, b in pairs
        if frozenset((a, b)) in documented
    ]
    doc_outcomes = outcomes.summarise_documented(documented_texts)

    # Channel C findings lead: DrugBank's own words outrank anything inferred.
    all_findings = doc_outcomes["findings"] + analysis["findings"]
    channels_fired = sorted({f["channel"] for f in all_findings})
    if any(f.get("outcomes") for f in analysis["findings"]):
        channels_fired.append("outcome_annotation")

    coverage = dict(analysis["coverage"])
    coverage["documented_pairs_without_named_outcome"] = doc_outcomes[
        "documented_pairs_without_named_outcome"
    ]

    predicted_ades = []
    for r in pair_results:
        pair_label = f"{r['name_a']} × {r['name_b']} — " if multi_pair else ""
        if r["score1"] is not None:
            predicted_ades.append(
                {"ade": f"{pair_label}Spectral + gradient-boosted-trees classifier (k-fold ROC-AUC 0.784)", "score": r["score1"]}
            )
        if r["score2"] is not None:
            predicted_ades.append(
                {"ade": f"{pair_label}Task-supervised GCN classifier (k-fold ROC-AUC 0.945)", "score": r["score2"]}
            )
    predicted_ades.sort(key=lambda x: x["score"], reverse=True)

    grounding = [item for r in pair_results for item in r["grounding"]]

    if multi_pair:
        explanation = " ".join(
            f"[{r['name_a']} × {r['name_b']}] {r['explanation']}" for r in pair_results
        )
    else:
        explanation = pair_results[0]["explanation"]

    status = max((r["status"] for r in pair_results), key=lambda s: STATUS_RANK[s])
    pairs_documented = sum(1 for r in pair_results if r["status"] == "documented")
    pairs_predicted = sum(1 for r in pair_results if r["status"] == "predicted")

    note_parts = [NOTE_BASE]
    if multi_pair:
        note_parts.append(NOTE_MULTI_PAIR)
    if analysis["findings"]:
        note_parts.append(NOTE_CONVERGENCE)
    note = " ".join(note_parts)

    timings: dict = {"documented_interaction_scan": doc_scan_ms, "profile_scan": profile_scan_ms}
    for r in pair_results:
        for k, v in r["timings"].items():
            timings[k] = round(timings.get(k, 0) + v, 1)
    timings["total"] = round((time.time() - t0) * 1000, 1)

    names = [id_to_name[d] for d in valid]

    return {
        "found": True,
        "error": None,
        "error_type": None,
        "drugs": names,
        "drug_ids": valid,
        "graph_file": None,
        "below_training_threshold": None,
        "predicted_ades": predicted_ades,
        "grounding": grounding,
        "explanation": explanation,
        "findings": all_findings,
        "channels_fired": channels_fired,
        "coverage": coverage,
        "report": {
            "interaction_status": status,
            "pairs_total": len(pair_results),
            "pairs_documented": pairs_documented,
            "pairs_predicted": pairs_predicted,
            "interaction_summary": explanation,
            "note": note,
        },
        "meta": {"request_id": None, "timings_ms": timings, "use_real_model": True},
    }


_drug_names_cache: list | None = None


@app.get("/drugs")
def drugs_endpoint():
    global _drug_names_cache
    if _drug_names_cache is None:
        _drug_names_cache = sorted(set(_state["id_to_name"].values()), key=str.lower)
    return {"drugs": _drug_names_cache}


@app.get("/health")
def health_endpoint():
    return {
        "status": "ok",
        "use_real_model": True,
        "model_loaded": _state.get("clf") is not None or _state.get("decoder") is not None,
        "model1_loaded": _state.get("clf") is not None,
        "model2_loaded": _state.get("decoder") is not None,
        "gap_filler_loaded": _state.get("mech_model") is not None,
        "drugs_indexed": len(_state.get("id_to_name", {})),
    }
