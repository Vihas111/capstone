import os
import sys
import time

CAPSTONE_ROOT = "/Volumes/Elements/Capstone"
os.chdir(CAPSTONE_ROOT)
sys.path.insert(0, CAPSTONE_ROOT)

t0 = time.time()
from scripts import mechanism_lookup as ml
from scripts import predict_multidrug_interaction as pmi

print("imports ok", time.time() - t0)

id_to_name, name_to_id = ml.load_drug_index()
print("drug index:", len(id_to_name), "drugs")

a = ml.resolve_drug("warfarin", id_to_name, name_to_id)
b = ml.resolve_drug("acetylsalicylic acid", id_to_name, name_to_id)
print("resolved:", a, b)

enzymes = ml.load_per_drug_table(ml.DATA_DIR / "drug_enzymes.jsonl", "enzyme", "drug", [a, b])
targets = ml.load_per_drug_table(ml.DATA_DIR / "drug_targets.jsonl", "target", "drug", [a, b])
transporters = ml.load_per_drug_table(ml.DATA_DIR / "drug_transporters.jsonl", "transporter", "drug", [a, b])
carriers = ml.load_per_drug_table(ml.DATA_DIR / "drug_carriers.jsonl", "carrier", "drug", [a, b])

documented = ml.find_documented_interactions({a, b})
print("documented:", documented.get(frozenset((a, b))))

overlap = ml.shared(enzymes[a], enzymes[b], id_to_name[a], id_to_name[b], infer_pk=True)
print("enzyme overlap:", overlap)

t1 = time.time()
clf, cache = pmi.load_model1(pmi.DEFAULT_HGB_MODEL, pmi.DEFAULT_HGB_CACHE)
print("model1 loaded", time.time() - t1)
score1 = pmi.score_pair_model1(clf, cache, a, b)
print("model1 score:", score1)

import torch
from scripts.run_pair_link_classifier_gnn_kfold import PairDecoder


def load_model2_cpu(embeddings_path, decoder_path):
    data = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    node_index, embeddings, out_dim = data["node_index"], data["embeddings"], data["out_dim"]
    decoder = PairDecoder(out_dim)
    decoder.load_state_dict(torch.load(decoder_path, map_location="cpu", weights_only=True))
    decoder.eval()
    return node_index, embeddings, decoder


t2 = time.time()
node_index, embeddings, decoder = load_model2_cpu(pmi.DEFAULT_GNN_EMBEDDINGS, pmi.DEFAULT_GNN_DECODER)
print("model2 loaded", time.time() - t2)
score2 = pmi.score_pair_model2(node_index, embeddings, decoder, a, b)
print("model2 score:", score2)

t3 = time.time()
model, label_vocab = ml.load_mechanism_model(
    ml.DEFAULT_PREDICT_CHECKPOINT, ml.DEFAULT_PREDICT_VOCAB, ml.DEFAULT_PREDICT_HIDDEN_DIMS
)
knn_pool = ml.load_knn_pool(ml.DEFAULT_KNN_FINGERPRINTS, ml.DEFAULT_KNN_DATASET, ml.DEFAULT_KNN_SPLIT)
thresholds = ml.load_label_thresholds(ml.DEFAULT_PREDICT_THRESHOLDS_BLEND, label_vocab)
smiles_by_drug = ml.load_drug_smiles()
print("gap-filler loaded", time.time() - t3)

pred_a = ml.predict_mechanisms(smiles_by_drug.get(a), model, label_vocab, thresholds, knn_pool=knn_pool)
print("predicted mechanisms for a:", {k: len(v) for k, v in (pred_a or {}).items()})

print("TOTAL", time.time() - t0)
