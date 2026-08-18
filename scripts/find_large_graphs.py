# =========================================================
# find_large_graphs.py
# =========================================================

import os
import json

PAIR_DIR = "data/pair_graphs"

print("Scanning graphs...")

results = []

files = [
    f for f in os.listdir(PAIR_DIR)
    if f.endswith(".json")
]

for i, file in enumerate(files):

    path = os.path.join(
        PAIR_DIR,
        file
    )

    with open(
        path,
        encoding="utf-8"
    ) as f:

        graph = json.load(f)

    results.append({

        "file": file,

        "nodes":
            len(graph["nodes"]),

        "edges":
            len(graph["edges"]),

        "ade_count":
            len(
                graph.get(
                    "ade_evidence",
                    []
                )
            )

    })

    if i % 10000 == 0 and i > 0:

        print(
            f"Processed {i:,}"
        )

# =========================================================
# SORT
# =========================================================

results.sort(
    key=lambda x: x["nodes"],
    reverse=True
)

# =========================================================
# TOP 20
# =========================================================

print("\n===== TOP 20 LARGEST GRAPHS =====\n")

for row in results[:20]:

    print(
        f"{row['file']:<35}"
        f" Nodes={row['nodes']:<6}"
        f" Edges={row['edges']:<6}"
        f" ADEs={row['ade_count']}"
    )

# =========================================================
# DISTRIBUTION
# =========================================================

sizes = [r["nodes"] for r in results]

print("\n===== SIZE DISTRIBUTION =====\n")

print("Graphs:", len(sizes))

print(
    "Average:",
    round(sum(sizes) / len(sizes), 2)
)

print("Max:", max(sizes))
print("Min:", min(sizes))

for threshold in [

    100,
    250,
    500,
    1000,
    2000

]:

    count = sum(

        1
        for s in sizes
        if s >= threshold

    )

    print(
        f">= {threshold:4}: "
        f"{count:,}"
    )

print("\nDone.")