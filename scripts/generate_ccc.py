#!/usr/bin/env python3
"""Generate dummy CCC addendum and figures for pipeline checks."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT_DIR = Path("evaluation/ccc/addenda")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

rows = [
    {"rail_a": "A", "rail_b": "B", "mode": "AND", "successes": 30, "trials": 50},
    {"rail_a": "C", "rail_b": "D", "mode": "OR", "successes": 40, "trials": 80},
    {"rail_a": "E", "rail_b": "F", "mode": "AND", "successes": 10, "trials": 40},
]

for r in rows:
    r["C_hat"] = r["successes"] / r["trials"] if r["trials"] else 0.0
    r["L"] = 0.0
    r["U"] = 1.0
    r["I"] = 0.5
    r["CCC"] = r["C_hat"]
    r["headroom"] = r["U"] - r["C_hat"]
    r["lift"] = r["C_hat"] - r["L"]

with (OUTPUT_DIR / "ccc_addendum.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "rail_a",
            "rail_b",
            "mode",
            "L",
            "I",
            "U",
            "successes",
            "trials",
            "C_hat",
            "CCC",
            "headroom",
            "lift",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1])
fig.savefig(OUTPUT_DIR / "example__AND.png")
plt.close(fig)
