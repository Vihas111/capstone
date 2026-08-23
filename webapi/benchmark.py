"""Validation benchmark for the multi-channel detector.

Because the detector is deterministic rules rather than a trained model, it is
validatable by EXPERT REVIEW of a small curated set rather than by a large
labelled dataset -- which is what makes validation tractable here at all.

Each case declares which channels SHOULD fire and, for dangerous combinations,
what the real-world concern is, so a domain expert can check the tool's reading
against their own without reading any code.

    ./.venv/bin/python benchmark.py            # run all
    ./.venv/bin/python benchmark.py --names    # just verify drug names resolve

A case "passes" only in the weak sense that the channels behaved as expected.
Whether the FINDING TEXT is pharmacologically sound is the part that needs a
human -- run with --verbose and read it.
"""

import argparse
import json
import sys
import urllib.request

API = "http://127.0.0.1:8000"

# expect_fire: channels that must produce at least one finding
# expect_quiet: True if the combination should produce no findings at all
CASES = [
    # ---- dangerous pairs ----
    dict(label="Warfarin + Aspirin", drugs=["Warfarin", "Acetylsalicylic acid"],
         concern="bleeding (anticoagulant potentiation)",
         expect_fire=["documented_outcome"]),
    dict(label="Sildenafil + Nitroglycerin", drugs=["Sildenafil", "Nitroglycerin"],
         concern="severe hypotension (contraindicated)",
         expect_fire=["documented_outcome"]),
    dict(label="Fluoxetine + Phenelzine", drugs=["Fluoxetine", "Phenelzine"],
         concern="serotonin syndrome (MAOI + SSRI, contraindicated)",
         expect_fire=["documented_outcome", "risk_axis"]),
    # DrugBank's text here is PK-only ("serum concentration of Digoxin can be
    # increased"), so Channel C legitimately stays quiet -- the outcome has to
    # come from Channel D annotating digoxin's own risk classes instead. This
    # case exists specifically to keep that division of labour honest.
    dict(label="Digoxin + Amiodarone", drugs=["Digoxin", "Amiodarone"],
         concern="digoxin toxicity via P-gp inhibition",
         expect_fire=["shared_protein", "outcome_annotation", "risk_axis"]),
    dict(label="Lithium + Ibuprofen", drugs=["Lithium carbonate", "Ibuprofen"],
         concern="reduced lithium clearance -> toxicity",
         expect_fire=[]),
    dict(label="Clarithromycin + Simvastatin", drugs=["Clarithromycin", "Simvastatin"],
         concern="raised statin exposure -> myopathy",
         expect_fire=["outcome_annotation"]),

    # ---- dangerous triples ----
    dict(label="Clarithromycin + Simvastatin + Amlodipine",
         drugs=["Clarithromycin", "Simvastatin", "Amlodipine"],
         concern="two CYP3A4 inhibitors stacking on a statin -> rhabdomyolysis",
         expect_fire=["shared_protein", "outcome_annotation"]),
    dict(label="Lisinopril + HCTZ + Ibuprofen (triple whammy)",
         drugs=["Lisinopril", "Hydrochlorothiazide", "Ibuprofen"],
         concern="acute kidney injury -- the organ-level case the protein channel misses",
         expect_fire=["risk_axis", "documented_outcome"]),
    dict(label="Warfarin + Aspirin + Clopidogrel",
         drugs=["Warfarin", "Acetylsalicylic acid", "Clopidogrel"],
         concern="triple antithrombotic -> major bleeding",
         expect_fire=["risk_axis", "documented_outcome"]),

    # ---- benign controls: should stay quiet ----
    dict(label="Amoxicillin + Vitamin C", drugs=["Amoxicillin", "Ascorbic acid"],
         concern=None, expect_quiet=True),
    dict(label="Loratadine + Amoxicillin", drugs=["Loratadine", "Amoxicillin"],
         concern=None, expect_quiet=True),
    dict(label="Vitamin C + Levothyroxine", drugs=["Ascorbic acid", "Levothyroxine"],
         concern=None, expect_quiet=True),
]


def post(path, payload, timeout=180):
    req = urllib.request.Request(
        API + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def check_names():
    with urllib.request.urlopen(API + "/drugs", timeout=30) as r:
        known = {n.lower() for n in json.loads(r.read())["drugs"]}
    bad = sorted({d for c in CASES for d in c["drugs"] if d.lower() not in known})
    if bad:
        print("UNRESOLVABLE DRUG NAMES (exact-match only):")
        for b in bad:
            print("   ", b)
        return False
    print(f"all {len({d for c in CASES for d in c['drugs']})} drug names resolve")
    return True


def run(verbose=False):
    passed = failed = 0
    for case in CASES:
        d = post("/predict", {"drugs": case["drugs"]})
        fired = set(d.get("channels_fired") or [])
        findings = d.get("findings") or []
        quiet_expected = case.get("expect_quiet", False)
        missing = [c for c in case.get("expect_fire", []) if c not in fired]

        if quiet_expected:
            ok = not findings
            detail = "no findings" if ok else f"UNEXPECTED: {sorted(fired)}"
        else:
            ok = not missing
            detail = f"fired={sorted(fired) or 'NONE'}" + (f"  MISSING={missing}" if missing else "")

        passed, failed = (passed + 1, failed) if ok else (passed, failed + 1)
        print(f"[{'PASS' if ok else 'FAIL'}] {case['label']}")
        print(f"        {detail}")
        if case.get("concern"):
            print(f"        real concern: {case['concern']}")
        if verbose:
            for f in findings:
                print(f"          - ({f['channel']}) {f['finding']}")
            cov = d.get("coverage") or {}
            for n in cov.get("notes", []):
                print(f"          ! {n}")
        print()

    print(f"{passed} passed, {failed} failed, {len(CASES)} total")
    return failed == 0


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--names", action="store_true", help="only verify drug names resolve")
    p.add_argument("--verbose", "-v", action="store_true", help="print every finding")
    a = p.parse_args()
    if a.names:
        sys.exit(0 if check_names() else 1)
    if not check_names():
        sys.exit(1)
    print()
    sys.exit(0 if run(a.verbose) else 1)
