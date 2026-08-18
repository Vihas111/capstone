# Sample outputs — scripts/mechanism_lookup.py

Three real outputs from the deterministic DrugBank mechanism-lookup tool,
covering the three cases the tool can produce:

- **case1_documented_interaction.json** — Warfarin + Aspirin (DB00682 +
  DB00945). Has a DrugBank-documented interaction AND mechanistic overlaps
  (shared CYP2C9/CYP2C19 enzymes, with rule-based PK readings).
- **case2_mechanistic_overlap_only.json** — Aldesleukin + Pegvisomant
  (DB00041 + DB00082). No documented DrugBank interaction, but they share
  CYP3A4 — the tool still surfaces this as a mechanistic overlap.
- **case3_no_signal.json** — Lepirudin + Cetuximab (DB00001 + DB00002).
  Neither a documented interaction nor any shared enzyme/target/transporter/
  carrier. The tool still returns a clean, complete report (per-drug
  profiles + an honest "none found") rather than erroring or omitting
  sections.

Each file has two top-level parts:
- `part_1_mechanism_report` — the raw DrugBank-sourced facts (no ML/LLM)
- `part_2_downstream_prompt` — a formatted prompt string, ready to hand to a
  downstream model alongside real patient data and RAG-retrieved literature
  (see the `{{PATIENT_DATA}}` / `{{RETRIEVED_CONTEXT}}` placeholders inside it)

Regenerate any of these, or try new drug combinations, with:
```bash
python scripts/mechanism_lookup.py DB00682 DB00945 --pretty
python scripts/mechanism_lookup.py warfarin "acetylsalicylic acid" --pretty   # names also work
```
