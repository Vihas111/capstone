"""Reference data loaded ONCE at startup, shared by every detection channel.

Everything here is read-only consumption of files the Capstone repo's
extract_biomedical_features.py already produced. Nothing in this module (or
anywhere else in webapi/) writes to that repo -- it is mounted read-only.

Two things justify loading these fully into memory rather than scanning per
request, as scripts/mechanism_lookup.py's load_per_drug_table() does:

  1. They are small. enzymes 6k lines, targets 24k, transporters 3.6k,
     carriers 1k, categories 107k -- together a few tens of MB. (The one
     genuinely large file, drugbank_interactions.jsonl at 463MB, is NOT
     loaded here; it stays a per-request scan in main.py.)
  2. Base-rate weighting NEEDS the whole population, not just the queried
     drugs. "How many of all 19,830 drugs touch this protein" cannot be
     computed from a filtered view, and without it the detector cannot tell
     an informative overlap (a rare transporter) from a meaningless one
     (albumin, which nearly every drug binds).
"""

import collections
import json
import re
from pathlib import Path

import orjson

DATA_DIR = Path("data/processed")
CONFIG_DIR = Path(__file__).resolve().parent

PROFILE_FILES = (
    ("enzyme", "drug_enzymes.jsonl", "enzyme"),
    ("target", "drug_targets.jsonl", "target"),
    ("transporter", "drug_transporters.jsonl", "transporter"),
    ("carrier", "drug_carriers.jsonl", "carrier"),
)

# Only genuinely overlapping isoforms are collapsed. CYP3A4/3A5/3A7 share
# substrates so heavily that reporting them separately produces three
# near-identical findings for one real mechanism (observed directly on the
# clarithromycin/simvastatin/amlodipine case). CYP2C9 vs CYP2C19 are NOT
# collapsed despite the similar naming -- they are clinically distinct
# enzymes with different substrate profiles, and merging them would be wrong.
PROTEIN_FAMILIES = {
    "CYP3A4": "CYP3A",
    "CYP3A5": "CYP3A",
    "CYP3A7": "CYP3A",
}

_CYP_FROM_PROTEIN = re.compile(r"cytochrome\s+p[-\s]?450\s+([0-9]+[a-z]+[0-9]*)", re.I)
_CYP_FROM_CATEGORY = re.compile(
    r"cytochrome\s+p[-\s]?450\s+(cyp[0-9]+[a-z]*[0-9]*)\s+(inhibitor|inducer)s?"
    r"(?:\s*\((strong|moderate|weak|strength unknown)\))?",
    re.I,
)


def _load_jsonl(path):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def normalize_protein(name: str) -> str | None:
    """'Cytochrome P450 3A4' -> 'CYP3A4'. Returns None for non-CYP proteins,
    which simply have no strength annotation available."""
    m = _CYP_FROM_PROTEIN.search(name or "")
    return f"CYP{m.group(1).upper()}" if m else None


def family_of(name: str) -> str | None:
    """Collapsed family label for a protein, or None if it stands alone."""
    cyp = normalize_protein(name)
    return PROTEIN_FAMILIES.get(cyp) if cyp else None


class Knowledge:
    def __init__(self):
        # {kind: {drug_id: [{"name":, "actions": [...]}, ...]}}
        self.profiles: dict[str, dict[str, list]] = {}
        # {(kind, protein_name): count of distinct drugs touching it}
        self.protein_drug_count: collections.Counter = collections.Counter()
        # {kind: total drugs having ANY entry of that kind}
        self.kind_drug_total: dict[str, int] = {}
        # {drug_id: {category, ...}}
        self.categories: dict[str, set[str]] = {}
        # {(cyp, role): {drug_id: strength}} e.g. ("CYP3A4","inhibitor") -> {"DB01211": "strong"}
        self.cyp_strength: dict[tuple, dict[str, str]] = collections.defaultdict(dict)

        self.risk_axes: list[dict] = []
        self.outcome_classes: dict[str, str] = {}

    def load(self, log=None):
        for kind, filename, field in PROFILE_FILES:
            by_drug: dict[str, list] = collections.defaultdict(list)
            seen_pairs = set()
            for row in _load_jsonl(DATA_DIR / filename):
                drug, protein = row["drug"], row[field]
                entry = {"name": protein}
                if "actions" in row:
                    entry["actions"] = row["actions"]
                by_drug[drug].append(entry)
                # count each (protein, drug) once even if the file repeats it
                key = (kind, protein, drug)
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    self.protein_drug_count[(kind, protein)] += 1
            self.profiles[kind] = dict(by_drug)
            self.kind_drug_total[kind] = len(by_drug)

        cats: dict[str, set] = collections.defaultdict(set)
        for row in _load_jsonl(DATA_DIR / "drug_categories.jsonl"):
            cat = row["category"]
            drug = row["drug"]
            cats[drug].add(cat)
            m = _CYP_FROM_CATEGORY.search(cat)
            if m:
                cyp, role, strength = m.group(1).upper(), m.group(2).lower(), m.group(3)
                strength = (strength or "").lower()
                if strength == "strength unknown":
                    strength = "unspecified"
                prev = self.cyp_strength[(cyp, role)].get(drug)
                # a drug can carry both '(weak)' and '(strength unknown)' tags for the
                # same enzyme; prefer a concrete grade over 'unspecified'
                if prev is None or (prev == "unspecified" and strength):
                    self.cyp_strength[(cyp, role)][drug] = strength or "unspecified"
        self.categories = {d: set(v) for d, v in cats.items()}

        self.risk_axes = json.loads((CONFIG_DIR / "risk_axes.json").read_text())["axes"]
        self.outcome_classes = json.loads(
            (CONFIG_DIR / "outcome_classes.json").read_text()
        )["outcome_classes"]

        if log:
            log.info(
                "Knowledge: %d enzyme / %d target / %d transporter / %d carrier profiles, "
                "%d drugs with categories, %d risk axes, %d outcome classes",
                len(self.profiles["enzyme"]), len(self.profiles["target"]),
                len(self.profiles["transporter"]), len(self.profiles["carrier"]),
                len(self.categories), len(self.risk_axes), len(self.outcome_classes),
            )

    # ---- per-query accessors -------------------------------------------------

    def profile(self, kind: str, drug_id: str) -> list:
        return self.profiles.get(kind, {}).get(drug_id, [])

    def commonness(self, kind: str, protein: str) -> float:
        """Fraction of drugs (among those with any data of this kind) that touch
        this protein. Albumin approaches 1.0; a niche transporter is near 0.
        Used to damp findings on proteins so common they carry no information."""
        total = self.kind_drug_total.get(kind) or 1
        return self.protein_drug_count.get((kind, protein), 0) / total

    def strength_for(self, drug_id: str, protein: str, role: str) -> str | None:
        """DrugBank's own strong/moderate/weak grading of this drug's inhibition
        or induction of this enzyme, when it publishes one. Only CYPs carry
        these tags; everything else returns None (meaning 'not graded', which
        callers must not render as 'weak')."""
        cyp = normalize_protein(protein)
        if not cyp:
            return None
        return self.cyp_strength.get((cyp, role), {}).get(drug_id)

    def outcome_labels(self, drug_id: str) -> list[str]:
        """Plain-English consequences of having MORE of this drug on board."""
        cats = self.categories.get(drug_id, ())
        return [self.outcome_classes[c] for c in cats if c in self.outcome_classes]

    def has_mechanism_data(self, drug_id: str) -> bool:
        return any(self.profile(k, drug_id) for k, _, _ in PROFILE_FILES)

    def has_category_data(self, drug_id: str) -> bool:
        return bool(self.categories.get(drug_id))


KB = Knowledge()
