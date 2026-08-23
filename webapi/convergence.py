"""Multi-channel deterministic interaction detector.

WHY MULTIPLE CHANNELS. The first version of this module had exactly one
detection channel -- proteins that every drug in the combination touches --
fed by one rule table over DrugBank action verbs. That design has a failure
mode worse than being wrong: when its single channel is blind, it returns
nothing, and "no findings" renders as "nothing to worry about".

That is not hypothetical. Benchmarked against the ACE-inhibitor + diuretic +
NSAID "triple whammy" (one of the most-cited real-world dangerous
combinations), it found nothing at all -- correctly, by its own logic, since
no single protein is touched by all three drugs. The convergence there is at
the ORGAN level: each drug independently reduces renal perfusion by a
different mechanism. A shared-protein detector cannot see that category of
harm no matter how well it is tuned.

So this module runs independent channels with DIFFERENT blind spots, and
reports coverage explicitly so that silence is distinguishable from safety:

  Channel A  shared-protein convergence     PK stacking on one enzyme/transporter
  Channel B  shared risk-axis convergence   additive same-direction effects,
                                            mechanism-independent (fixes the above)
  Channel D  outcome annotation             names the consequence of A's findings

(Channel C, extracting named outcomes from DrugBank's own interaction text,
lives in outcomes.py since it operates on interaction rows rather than
per-drug profiles.)

Every finding carries `channel` provenance, and every response carries a
coverage report saying which drugs each channel could actually speak for.

HONESTY BOUNDARY, unchanged from the single-channel version: these are
deterministic rules over DrugBank annotations -- no ML, no LLM -- and a
rule-based reading of action verbs and category tags is not a clinical
conclusion. Nothing here models dose, timing, or the patient.
"""

from itertools import combinations

from knowledge import KB, family_of

# A protein touched by more than this fraction of the population carries little
# information -- nearly every drug binds albumin, and roughly half are CYP3A4
# substrates, so "they all touch it" is unremarkable. Findings on such proteins
# are still reported when the ACTION PATTERN is risky (2+ inhibitors stacking on
# CYP3A4 is real and common), but bare overlap on them is suppressed.
COMMONNESS_SUPPRESS = 0.25

# Carriers are excluded from action-pattern inference: DrugBank rarely records
# actions for them, and the overwhelmingly common one (albumin) is noise.
INFERENCE_KINDS = ("enzyme", "transporter", "target")

STRENGTH_ORDER = {"strong": 3, "moderate": 2, "unspecified": 1, "weak": 0}


# ---------------------------------------------------------------------------
# Channel A -- shared-protein convergence
# ---------------------------------------------------------------------------

def _roles_on_protein(kind, protein, drug_ids, id_to_name):
    """Split the drugs touching one protein into perpetrators and victims.

    A drug can be BOTH, and getting this wrong breaks real cases in opposite
    directions. Digoxin is a P-gp substrate *and* inhibitor; classifying it as
    a perpetrator only (an earlier version did) makes it impossible to
    recognise as the victim it actually is when amiodarone inhibits the same
    transporter -- a textbook interaction the detector silently missed.
    Conversely, treating every substrate as a victim makes clarithromycin and
    amlodipine look like they are "cleared through" CYP3A4 alongside
    simvastatin, burying which drug actually accumulates.

    So: perpetrators and victims are computed independently and may overlap.
    Callers pair each victim with perpetrators OTHER THAN ITSELF, and prefer
    pure substrates (no inhibitor/inducer action) as the clearest victims."""
    perpetrators, victims = [], []
    for d in drug_ids:
        entry = next((e for e in KB.profile(kind, d) if e["name"] == protein), None)
        if entry is None:
            continue
        actions = set(entry.get("actions", []))
        rec = {"drug_id": d, "name": id_to_name.get(d, d), "actions": sorted(actions)}
        if "inhibitor" in actions or "inducer" in actions:
            role = "inhibitor" if "inhibitor" in actions else "inducer"
            p = dict(rec, role=role, strength=KB.strength_for(d, protein, role))
            perpetrators.append(p)
        if "substrate" in actions:
            victims.append(dict(rec, pure=not (actions & {"inhibitor", "inducer"})))
    return perpetrators, victims


def _describe_strength(perps):
    bits = []
    for p in perps:
        s = p.get("strength")
        bits.append(f"{p['name']} ({s})" if s and s != "unspecified" else p["name"])
    return ", ".join(bits)


def _channel_a(drug_ids, id_to_name):
    """Fires when at least one drug's clearance through a protein is affected by
    another drug in the combination.

    Two perpetrators or more is the genuinely N-way signal -- pairwise
    comparison can never see stacking, because each pair contains only one
    perpetrator. But a SINGLE perpetrator raising a victim's exposure is the
    most common dangerous PK pattern there is (clarithromycin + simvastatin),
    so excluding it -- as an earlier version did by requiring 2+ -- silently
    missed textbook interactions.

    Single-perpetrator findings are gated on being able to say why they matter:
    either the perpetrator's inhibition is graded strong/moderate by DrugBank,
    or the victim carries an outcome class. "Concentration changes" with no
    consequence attached is exactly the vacuous output this work set out to fix,
    and reporting every such pair would bury the real findings.

    Deliberately NOT requiring all N drugs to touch the protein: in a 4-drug
    combination a dangerous 3-drug convergence is still dangerous, and the
    converging subset is what should be reported."""
    findings = []
    grouped: dict[tuple, list] = {}

    for kind in INFERENCE_KINDS:
        proteins = {e["name"] for d in drug_ids for e in KB.profile(kind, d)}
        for protein in proteins:
            perps, victims = _roles_on_protein(kind, protein, drug_ids, id_to_name)
            if not perps or not victims:
                continue

            # prefer pure substrates as victims; fall back to mutual ones
            # (substrate AND inhibitor, e.g. digoxin on P-gp)
            pure = [v for v in victims if v["pure"]]

            if pure:
                # pure victims never inhibit, so victim and perpetrator sets are
                # disjoint by construction and can be reported together
                victim_recs = pure
                victim_ids = {v["drug_id"] for v in victim_recs}
                perp_list = [p for p in perps if p["drug_id"] not in victim_ids]
            else:
                # Mutual inhibition: every substrate here also inhibits. Unioning
                # them produced sentences naming the same drug as both victim and
                # perpetrator ("Clarithromycin, Amlodipine is cleared through
                # ABCB1, which ... Amlodipine, Clarithromycin inhibit"). Report a
                # single victim against only its external perpetrators instead.
                best = max(
                    victims,
                    key=lambda v: len([p for p in perps if p["drug_id"] != v["drug_id"]]),
                )
                victim_recs = [best]
                perp_list = [p for p in perps if p["drug_id"] != best["drug_id"]]

            if not perp_list:
                continue

            if len(perp_list) < 2:
                strengths = {(p.get("strength") or "ungraded") for p in perp_list}
                has_outcome = any(KB.outcome_labels(v["drug_id"]) for v in victim_recs)
                # DrugBank explicitly grading an inhibitor 'weak' is a statement
                # that it does not matter much; taking it as a finding produced a
                # real false positive (amoxicillin as a weak CYP2C8 inhibitor
                # "raising loratadine exposure"). Only strong/moderate stands on
                # its own; ungraded needs the victim to have a named consequence.
                meaningful = bool(strengths & {"strong", "moderate"}) or (
                    not (strengths & {"weak"}) and has_outcome
                )
                if not meaningful:
                    continue

            # NOTE: commonness suppression deliberately happens AFTER merging,
            # not here -- see the filter below the merge step for why.

            # collapse isoform families (CYP3A4/3A5) into one finding
            key = (kind, family_of(protein) or protein)
            grouped.setdefault(key, []).append((protein, perp_list, victim_recs))

    # Merge entries that describe the SAME drugs acting on each other, differing
    # only in which protein carries it. Amiodarone inhibits both BSEP and P-gp,
    # each raising digoxin exposure -- reporting those as two findings with
    # identical "why that matters" text is duplication, not extra information.
    merged: dict[tuple, list] = {}
    for (kind, label), members in grouped.items():
        members.sort(key=lambda m: -len(m[1]))
        protein, perps, victims = members[0]
        also = sorted({m[0] for m in members if m[0] != protein})
        key = (
            kind,
            frozenset(v["drug_id"] for v in victims),
            frozenset(p["drug_id"] for p in perps),
            frozenset(p["role"] for p in perps),
        )
        merged.setdefault(key, []).append((label, [protein] + also, perps, victims))

    for (kind, _vk, _pk, _rk), entries in merged.items():
        entries.sort(key=lambda e: -len(e[2]))
        label = " / ".join(e[0] for e in entries)
        proteins = [p for e in entries for p in e[1]]
        perps, victims = entries[0][2], entries[0][3]
        perps.sort(key=lambda p: -STRENGTH_ORDER.get(p.get("strength") or "unspecified", 1))

        # Drop findings carried ENTIRELY by proteins so common they say nothing
        # (over half of all drugs touch P-gp/ABCB1), unless DrugBank grades a
        # perpetrator as a meaningful inhibitor. Applied post-merge on purpose:
        # filtering per-protein beforehand discarded ABCB1 from the
        # digoxin/amiodarone finding and left it attributed to the far less
        # canonical bile salt export pump. Merging first keeps P-gp named
        # alongside its siblings while still dropping findings that rest on a
        # common protein alone.
        rarest = min(KB.commonness(kind, p) for p in proteins)
        graded = any((p.get("strength") or "") in ("strong", "moderate") for p in perps)
        if rarest > COMMONNESS_SUPPRESS and not graded:
            continue

        directions = {p["role"] for p in perps}
        victim_names = ", ".join(v["name"] for v in victims)
        stacking = len(perps) >= 2

        if directions == {"inhibitor"}:
            if stacking:
                effect = (
                    f"{victim_names} is cleared through {label}, which {len(perps)} other "
                    f"drugs in this combination inhibit ({_describe_strength(perps)}). "
                    f"Their inhibition stacks, so {victim_names} exposure rises more than "
                    f"any single pair suggests."
                )
            else:
                effect = (
                    f"{victim_names} is cleared through {label}, which "
                    f"{_describe_strength(perps)} inhibits, raising {victim_names} exposure."
                )
        elif directions == {"inducer"}:
            verb = "induce" if stacking else "induces"
            tail = (
                "Their induction stacks, potentially lowering "
                if stacking else "This potentially lowers "
            )
            effect = (
                f"{victim_names} is cleared through {label}, which "
                f"{_describe_strength(perps)} {verb}. {tail}{victim_names} exposure and efficacy."
            )
        else:
            effect = (
                f"{label} is both inhibited and induced by different drugs here "
                f"({_describe_strength(perps)}) while {victim_names} is cleared "
                f"through it. The net effect on exposure is opposing and cannot be "
                f"resolved by these rules -- needs expert review."
            )

        findings.append({
            "channel": "shared_protein",
            "kind": kind,
            "label": label,
            "proteins": proteins,
            "perpetrators": perps,
            "victims": victims,
            "commonness": round(min(KB.commonness(kind, p) for p in proteins), 3),
            "finding": effect,
            "outcomes": [],  # filled by Channel D
        })

    findings.sort(key=lambda f: (-len(f["perpetrators"]), f["commonness"]))
    return findings


# ---------------------------------------------------------------------------
# Channel B -- shared risk-axis convergence
# ---------------------------------------------------------------------------

def _channel_b(drug_ids, id_to_name):
    """Fires when 2+ drugs share an additive adverse-effect axis, regardless of
    whether they share any protein. This is the channel that sees organ-level
    convergence -- the triple-whammy case, where the three drugs share no
    protein but two of them are nephrotoxic and two raise potassium."""
    findings = []
    for axis in KB.risk_axes:
        cats = set(axis["categories"])
        tiers = axis.get("tiers", {})
        members = []
        for d in drug_ids:
            hit = cats & KB.categories.get(d, set())
            if not hit:
                continue
            tier = next((tiers[c] for c in hit if c in tiers), None)
            members.append({
                "drug_id": d,
                "name": id_to_name.get(d, d),
                "categories": sorted(hit),
                "tier": tier,
            })
        if len(members) < 2:
            continue

        listed = ", ".join(
            f"{m['name']} ({m['tier']})" if m["tier"] else m["name"] for m in members
        )
        findings.append({
            "channel": "risk_axis",
            "axis_id": axis["id"],
            "label": axis["label"],
            "drugs": members,
            "finding": (
                f"{len(members)} drugs in this combination independently contribute to "
                f"{axis['label']} ({listed}) -- {axis['consequence']}. They need not share "
                f"any protein or have a documented interaction for these effects to add up."
            ),
        })

    findings.sort(key=lambda f: -len(f["drugs"]))
    return findings


# ---------------------------------------------------------------------------
# Channel D -- outcome annotation (enriches Channel A)
# ---------------------------------------------------------------------------

MAX_OUTCOMES_PER_DRUG = 3


def _dedupe_labels(labels):
    """Collapse tiered variants of one class. A drug tagged both 'prolonging the
    QT interval' and 'prolonging the QT interval (moderate-risk group)' should
    read as one concern, not two -- keeping the longer (more specific) form."""
    kept: list[str] = []
    for lab in sorted(labels, key=len, reverse=True):
        if not any(k.startswith(lab.split(" (")[0]) for k in kept):
            kept.append(lab)
    return kept


def _annotate_outcomes(findings):
    """Channel A says exposure to a drug rises; this says why that matters.
    Without it the report stops at 'serum concentration increases', which is
    the specific complaint that motivated this work -- true, but it never
    names the actual harm.

    Annotates each victim only ONCE, on its highest-ranked finding. Repeating
    "Simvastatin is flagged as causing muscle toxicity" on every protein that
    raises simvastatin exposure is the same fact restated, and it drowns the
    findings that differ."""
    seen_victims: set[str] = set()
    for f in findings:
        if f["channel"] != "shared_protein":
            continue
        rising = f["victims"] if any(p["role"] == "inhibitor" for p in f["perpetrators"]) else []
        outcomes = []
        for v in rising:
            if v["drug_id"] in seen_victims:
                continue
            labels = _dedupe_labels(KB.outcome_labels(v["drug_id"]))[:MAX_OUTCOMES_PER_DRUG]
            if labels:
                seen_victims.add(v["drug_id"])
            for label in labels:
                outcomes.append({"drug": v["name"], "consequence": label})
        f["outcomes"] = outcomes
        if outcomes:
            by_drug: dict[str, list[str]] = {}
            for o in outcomes:
                by_drug.setdefault(o["drug"], []).append(o["consequence"])
            named = "; ".join(f"{d} is flagged as {', '.join(c)}" for d, c in by_drug.items())
            f["finding"] = f"{f['finding']} Why that matters: {named}."
    return findings


# ---------------------------------------------------------------------------
# Coverage -- makes silence legible
# ---------------------------------------------------------------------------

def _coverage(drug_ids, id_to_name):
    """Which channels could actually speak for which drugs. Without this, a
    drug with no DrugBank profile is silently indistinguishable from a drug
    that was checked and came back clean."""
    no_mech = [id_to_name.get(d, d) for d in drug_ids if not KB.has_mechanism_data(d)]
    no_cats = [id_to_name.get(d, d) for d in drug_ids if not KB.has_category_data(d)]
    notes = []
    if no_mech:
        notes.append(
            f"No enzyme/transporter/target data for {', '.join(no_mech)} -- the shared-protein "
            f"channel could not assess {'them' if len(no_mech) > 1 else 'it'}."
        )
    if no_cats:
        notes.append(
            f"No category data for {', '.join(no_cats)} -- the risk-axis channel could not "
            f"assess {'them' if len(no_cats) > 1 else 'it'}."
        )
    return {
        "drugs_without_mechanism_data": no_mech,
        "drugs_without_category_data": no_cats,
        "notes": notes,
    }


# ---------------------------------------------------------------------------

def analyse(drug_ids: list[str], id_to_name: dict, min_drugs: int = 2) -> dict:
    """Run every channel over the combination. Returns findings plus coverage.

    Runs for 2 drugs as well as 3+: Channel B (two drugs both nephrotoxic) and
    Channel D (exposure rises for a muscle-toxic drug) are meaningful for a
    plain pair, even though Channel A's stacking pattern needs 3+ to fire."""
    if len(drug_ids) < min_drugs:
        return {"findings": [], "coverage": _coverage(drug_ids, id_to_name), "channels_fired": []}

    findings = _annotate_outcomes(_channel_a(drug_ids, id_to_name)) + _channel_b(drug_ids, id_to_name)
    channels = sorted({f["channel"] for f in findings})
    if any(f.get("outcomes") for f in findings):
        channels.append("outcome_annotation")

    return {
        "findings": findings,
        "coverage": _coverage(drug_ids, id_to_name),
        "channels_fired": channels,
    }


CONVERGENCE_CAVEAT = (
    "Findings are deterministic rules over DrugBank annotations, not clinical conclusions. "
    "Dose, timing and patient factors are not modelled."
)
