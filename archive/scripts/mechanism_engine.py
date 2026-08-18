from interaction_index import InteractionIndex


class MechanismEngine:

    def __init__(self):

        self.index = InteractionIndex()

    # -----------------------------------
    # MAIN ENTRY
    # -----------------------------------

    def analyze_interaction(self, drug1, drug2):

        db1 = self.index.resolve_drug(drug1)
        db2 = self.index.resolve_drug(drug2)

        if not db1 or not db2:

            return {
                "error": "Drug not found"
            }

        results = {
            "drug1": db1,
            "drug2": db2,
            "enzyme_conflicts": [],
            "transporter_conflicts": [],
            "shared_targets": []
        }

        # -----------------------------------
        # ENZYME CONFLICTS
        # -----------------------------------

        enz1 = self.index.drug_to_enzymes[db1]
        enz2 = self.index.drug_to_enzymes[db2]

        for e1 in enz1:

            for e2 in enz2:

                if (
                    e1["enzyme"]
                    and
                    e1["enzyme"] == e2["enzyme"]
                ):

                    mechanisms = self._enzyme_logic(
                        e1["actions"],
                        e2["actions"]
                    )

                    if mechanisms:

                        results[
                            "enzyme_conflicts"
                        ].append({

                            "enzyme": e1["enzyme"],

                            "drug1_actions":
                                e1["actions"],

                            "drug2_actions":
                                e2["actions"],

                            "mechanisms":
                                mechanisms
                        })

        # -----------------------------------
        # TRANSPORTER CONFLICTS
        # -----------------------------------

        tr1 = self.index.drug_to_transporters[db1]
        tr2 = self.index.drug_to_transporters[db2]

        for t1 in tr1:

            for t2 in tr2:

                if (
                    t1["transporter"]
                    and
                    t1["transporter"] == t2["transporter"]
                ):

                    mechanisms = self._transporter_logic(
                        t1["actions"],
                        t2["actions"]
                    )

                    if mechanisms:

                        results[
                            "transporter_conflicts"
                        ].append({

                            "transporter":
                                t1["transporter"],

                            "drug1_actions":
                                t1["actions"],

                            "drug2_actions":
                                t2["actions"],

                            "mechanisms":
                                mechanisms
                        })

        # -----------------------------------
        # SHARED TARGETS
        # -----------------------------------

        targets1 = set(
            self.index.drug_to_targets[db1]
        )

        targets2 = set(
            self.index.drug_to_targets[db2]
        )

        shared = targets1 & targets2

        results["shared_targets"] = list(shared)

        return results

    # -----------------------------------
    # ENZYME REASONING
    # -----------------------------------

    def _enzyme_logic(self, a1, a2):

        findings = []

        # inhibitor + substrate
        if (
            "inhibitor" in a1
            and
            "substrate" in a2
        ):

            findings.append(
                "Drug1 may increase Drug2 exposure"
            )

        if (
            "substrate" in a1
            and
            "inhibitor" in a2
        ):

            findings.append(
                "Drug2 may increase Drug1 exposure"
            )

        # inducer + substrate
        if (
            "inducer" in a1
            and
            "substrate" in a2
        ):

            findings.append(
                "Drug1 may reduce Drug2 efficacy"
            )

        if (
            "substrate" in a1
            and
            "inducer" in a2
        ):

            findings.append(
                "Drug2 may reduce Drug1 efficacy"
            )

        # substrate competition
        if (
            "substrate" in a1
            and
            "substrate" in a2
        ):

            findings.append(
                "Potential metabolic competition"
            )

        return findings

    # -----------------------------------
    # TRANSPORTER REASONING
    # -----------------------------------

    def _transporter_logic(self, a1, a2):

        findings = []

        if (
            "inhibitor" in a1
            and
            "substrate" in a2
        ):

            findings.append(
                "Drug1 may alter Drug2 transport"
            )

        if (
            "substrate" in a1
            and
            "inhibitor" in a2
        ):

            findings.append(
                "Drug2 may alter Drug1 transport"
            )

        if (
            "substrate" in a1
            and
            "substrate" in a2
        ):

            findings.append(
                "Potential transporter competition"
            )

        return findings


# -----------------------------------
# TEST
# -----------------------------------

if __name__ == "__main__":

    engine = MechanismEngine()

    result = engine.analyze_interaction(
        "warfarin",
        "fluconazole"
    )

    import pprint
    pprint.pprint(result)