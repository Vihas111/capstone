"""Channel C -- named outcomes extracted from DrugBank's own interaction text.

The highest-authority answer to "what is the actual problem", because it is
DrugBank's own words rather than anything inferred. Roughly half of DrugBank's
interaction descriptions name a clinical outcome directly:

    "The risk or severity of serotonin syndrome can be increased when..."
    "The risk or severity of renal failure, hyperkalemia, and hypertension..."

and roughly half stop at pharmacokinetics, naming no outcome at all:

    "The serum concentration of Simvastatin can be increased when..."

That second shape is exactly the gap Channel D (outcome annotation, in
convergence.py) exists to fill. The two are complementary by construction:
C reports what DrugBank states, D infers a consequence only when DrugBank
stated none.

Extraction is a small set of anchored regexes over a fixed sentence grammar,
not an LLM and not a keyword bag. DrugBank's descriptions are template-
generated, so the templates are enumerable and matching them exactly is both
more precise than keyword spotting and honest about its own coverage -- text
that matches no template yields nothing rather than a guess.
"""

import re

# DrugBank's outcome-bearing templates, tagged by how the outcome should be
# phrased back. "risk" templates name a complication ("serotonin syndrome");
# "activity" templates name a pharmacological effect being amplified
# ("anticoagulant activities"), which reads wrong in a risk-of sentence.
_PATTERNS = (
    ("risk", re.compile(r"risk or severity of (?:the )?(.+?) can be (?:increased|decreased)", re.I)),
    ("activity", re.compile(r"may increase the (.+?) activities of", re.I)),
    ("activity", re.compile(r"may decrease the (.+?) activities of", re.I)),
)

# DrugBank's generic filler. "adverse effects" names nothing -- surfacing it as
# a finding would be noise dressed up as information.
_VACUOUS = {"adverse effects", "side effects", "adverse events", "adverse reactions",
            "toxicity", "side-effects"}

# Purely pharmacokinetic templates -- deliberately NOT treated as outcomes.
# "serum concentration increases" is a mechanism, not a consequence; letting it
# through would defeat the point of separating C from D.
_PK_ONLY = re.compile(
    r"^(the )?(serum concentration|metabolism|excretion|absorption|therapeutic efficacy|"
    r"bioavailability|protein binding)\b",
    re.I,
)

_SPLIT = re.compile(r",\s*and\s+|,\s*|\s+and\s+", re.I)


def extract_outcomes(text: str) -> list[tuple[str, str]]:
    """Named clinical outcomes stated in one DrugBank interaction description,
    as (outcome, kind) where kind is 'risk' or 'activity'.

    Empty list means DrugBank named none -- NOT that none exists."""
    if not text:
        return []

    out: list[tuple[str, str]] = []
    for kind, pat in _PATTERNS:
        for m in pat.finditer(text):
            phrase = m.group(1).strip().rstrip(".")
            if not phrase or _PK_ONLY.match(phrase):
                continue
            for part in _SPLIT.split(phrase):
                part = part.strip().rstrip(".").lower()
                if part in _VACUOUS:
                    continue
                # guard against runaway captures
                if part and 2 < len(part) < 60:
                    out.append((part, kind))

    seen, uniq = set(), []
    for o in out:
        if o[0] not in seen:
            seen.add(o[0])
            uniq.append(o)
    return uniq


def summarise_documented(pair_texts: list[tuple[str, str, str]]) -> dict:
    """pair_texts: [(name_a, name_b, description), ...] for pairs DrugBank
    documents. Returns the union of named outcomes plus which pairs named them,
    and how many documented pairs named nothing (the honest coverage figure --
    a high count means Channel C was largely silent for this combination and
    the other channels are carrying the weight)."""
    by_outcome: dict[tuple[str, str], list[str]] = {}
    pk_only_pairs: list[str] = []

    for name_a, name_b, text in pair_texts:
        outcomes = extract_outcomes(text)
        pair = f"{name_a} + {name_b}"
        if not outcomes:
            pk_only_pairs.append(pair)
            continue
        for o in outcomes:
            by_outcome.setdefault(o, []).append(pair)

    findings = []
    for (outcome, kind), pairs in sorted(by_outcome.items(), key=lambda kv: -len(kv[1])):
        where = "; ".join(pairs)
        if kind == "activity":
            sentence = (
                f"DrugBank states directly that this combination increases {outcome} "
                f"activity ({where})."
            )
        else:
            sentence = (
                f"DrugBank states directly that this combination raises the risk or "
                f"severity of {outcome} ({where})."
            )
        findings.append({
            "channel": "documented_outcome",
            "outcome": outcome,
            "pairs": pairs,
            "finding": sentence,
        })

    return {
        "findings": findings,
        "documented_pairs_without_named_outcome": pk_only_pairs,
    }
