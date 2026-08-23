import json
import time
import urllib.request

CASES = [
    # (label, drugs, expected)
    ("Warfarin x Aspirin", ["Warfarin", "Acetylsalicylic acid"], "DANGEROUS (bleeding, anticoagulant potentiation)"),
    ("Sildenafil x Nitroglycerin", ["Sildenafil", "Nitroglycerin"], "DANGEROUS (severe hypotension, contraindicated)"),
    ("Fluoxetine x Phenelzine", ["Fluoxetine", "Phenelzine"], "DANGEROUS (serotonin syndrome, MAOI+SSRI contraindicated)"),
    ("Digoxin x Amiodarone", ["Digoxin", "Amiodarone"], "DANGEROUS (digoxin toxicity via P-gp)"),
    ("Lithium x Ibuprofen", ["Lithium carbonate", "Ibuprofen"], "DANGEROUS (reduced lithium clearance -> toxicity)"),
    ("Amoxicillin x Acetaminophen", ["Amoxicillin", "Acetaminophen"], "NEGATIVE control (commonly co-prescribed, benign)"),
    ("Vitamin C x Ibuprofen", ["Ascorbic acid", "Ibuprofen"], "NEGATIVE control (unrelated mechanisms)"),
    ("Clarithromycin x Simvastatin x Amlodipine", ["Clarithromycin", "Simvastatin", "Amlodipine"], "DANGEROUS (CYP3A4 stacking -> myopathy/rhabdo risk)"),
    ("Lisinopril x HCTZ x Ibuprofen", ["Lisinopril", "Hydrochlorothiazide", "Ibuprofen"], "DANGEROUS ('triple whammy' acute kidney injury)"),
    ("Warfarin x Aspirin x Clopidogrel", ["Warfarin", "Acetylsalicylic acid", "Clopidogrel"], "DANGEROUS (triple antithrombotic, major bleeding risk)"),
    ("Amoxicillin x Acetaminophen x Vitamin C", ["Amoxicillin", "Acetaminophen", "Ascorbic acid"], "NEGATIVE control"),
]

results = []
for label, drugs, expected in CASES:
    t0 = time.time()
    req = urllib.request.Request(
        "http://127.0.0.1:8000/predict",
        data=json.dumps({"drugs": drugs}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            d = json.loads(resp.read())
    except Exception as e:
        print(f"{label}: ERROR {e}")
        continue
    ms = round((time.time() - t0) * 1000)

    top_scores = sorted([a["score"] for a in d.get("predicted_ades", [])], reverse=True)
    doc_any = any(g["section"] == "documented interaction" for g in d.get("grounding", []))
    n_way = d.get("n_way_convergence", []) or []
    n_way_findings = [e for e in n_way if e.get("finding")]

    results.append({
        "label": label,
        "expected": expected,
        "n_drugs": len(drugs),
        "found": d.get("found"),
        "severity": d.get("report", {}).get("clinical_severity") if d.get("found") else d.get("error"),
        "top_scores": top_scores,
        "documented": doc_any,
        "n_way_findings": len(n_way_findings),
        "ms": ms,
        "explanation": d.get("explanation"),
    })
    print(f"done: {label} ({ms}ms)")

json.dump(results, open("/tmp/batch_results.json", "w"), indent=2)
print("\n\nALL DONE")
