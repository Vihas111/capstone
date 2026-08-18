# =========================================================
# group_ades.py
#
# Purpose:
#   Automatically group common TWOSIDES ADEs
#   into broader clinical categories
#
# Input:
#   ade_frequencies.json
#
# Output:
#   grouped_ades.json
# =========================================================

import json

# =========================================================
# LOAD ADE FREQUENCIES
# =========================================================

with open("ade_frequencies.json") as f:
    ade_freq = json.load(f)

# =========================================================
# CLINICAL GROUP RULES
# =========================================================

GROUP_RULES = {

    "bleeding": [
        "hemorrhage",
        "bleeding",
        "hematoma",
        "melena",
        "hematemesis",
        "anaemia",
        "haemoglobin",
    ],

    "nephrotoxicity": [
        "renal",
        "kidney",
        "creatinine",
    ],

    "hepatotoxicity": [
        "hepatic",
        "liver",
        "bilirubin",
    ],

    "cns_depression": [
        "somnolence",
        "sedation",
        "loss of consciousness",
        "confusional",
        "dizziness",
        "syncope",
    ],

    "neurologic": [
        "headache",
        "tremor",
        "paraesthesia",
        "hypoaesthesia",
        "gait",
        "convulsion",
        "seizure",
    ],

    "psychiatric": [
        "anxiety",
        "depression",
        "insomnia",
    ],

    "arrhythmia": [
        "atrial fibrillation",
        "tachycardia",
        "bradycardia",
        "arrhythmia",
        "palpitations",
        "qt",
    ],

    "cardiovascular": [
        "myocardial infarction",
        "hypertension",
        "hypotension",
        "cardiac failure",
        "chest pain",
    ],

    "thrombotic": [
        "deep vein thrombosis",
        "pulmonary embolism",
        "thrombosis",
    ],

    "respiratory": [
        "dyspnoea",
        "respiratory",
        "cough",
        "pleural effusion",
    ],

    "infection": [
        "pneumonia",
        "sepsis",
        "infection",
        "pyrexia",
        "chills",
    ],

    "gastrointestinal": [
        "nausea",
        "vomiting",
        "diarrhoea",
        "abdominal pain",
        "constipation",
        "dysphagia",
    ],

    "dermatologic": [
        "rash",
        "pruritus",
        "erythema",
        "hyperhidrosis",
    ],

    "musculoskeletal": [
        "arthralgia",
        "myalgia",
        "muscle spasms",
        "muscular weakness",
        "back pain",
        "pain in extremity",
    ],

    "constitutional": [
        "fatigue",
        "malaise",
        "asthenia",
        "pain",
        "feeling abnormal",
        "dehydration",
        "fall",
        "death",
    ],

    "metabolic": [
        "weight",
        "glucose",
        "decreased appetite",
    ],

    "edema": [
        "oedema",
        "edema",
    ],

    "sensory": [
        "vision blurred",
    ],

    "hematologic": [
    "neutropenia",
    "thrombocytopenia",
    "pancytopenia",
    "leukopenia",
    "platelet",
    "white blood cell",
    ],

    "electrolyte": [
        "hypokalaemia",
        "hyponatraemia",
        "hyperkalaemia",
    ],

    "allergic": [
        "hypersensitivity",
        "urticaria",
        "drug hypersensitivity",
    ],

    "cognitive": [
        "amnesia",
        "memory impairment",
        "balance disorder",
    ],

    "pulmonary_disease": [
        "asthma",
        "chronic obstructive pulmonary disease",
        "copd",
        "bronchitis",
    ],

    "gastrointestinal": [
        "dyspepsia",
        "gastrooesophageal reflux",
        "abdominal discomfort",
        "abdominal distension",
    ],

    "cardiac": [
        "cardiac arrest",
        "cardiac disorder",
        "heart rate increased",
        "chest discomfort",
    ],

    "cerebrovascular": [
        "cerebrovascular accident",
    ],

    "medication_related": [
        "drug ineffective",
        "drug interaction",
        "off label use",
        "overdose",
        "condition aggravated",
    ]
}

# =========================================================
# GROUPING
# =========================================================

grouped = {}

ungrouped = []

TOP_K = 1000

top_ades = list(ade_freq.keys())[:TOP_K]

for ade in top_ades:

    matched = False

    for group, keywords in GROUP_RULES.items():

        for keyword in keywords:

            if keyword in ade:

                grouped[ade] = group

                matched = True

                break

        if matched:
            break

    if not matched:
        grouped[ade] = "other"
        ungrouped.append(ade)

# =========================================================
# SAVE OUTPUT
# =========================================================

with open("grouped_ades.json", "w") as f:

    json.dump(grouped, f, indent=2)

# =========================================================
# STATS
# =========================================================

print("\n===== GROUP COUNTS =====\n")

group_counts = {}

for ade, group in grouped.items():

    group_counts[group] = (
        group_counts.get(group, 0) + 1
    )

for group, count in sorted(
    group_counts.items(),
    key=lambda x: x[1],
    reverse=True
):

    print(f"{group:<25} {count}")

# =========================================================
# SHOW UNGROUPED
# =========================================================

print("\n===== SAMPLE UNGROUPED ADEs =====\n")

for ade in ungrouped[:50]:
    print(ade)

print("\nDone.")