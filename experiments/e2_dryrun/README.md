# E2 dry run

A rehearsal of the [E2 measurement contract](../../docs/research/E2_MEASUREMENT_CONTRACT.md)
pipeline against three toy guardrail mechanisms. **Not E2.** It validates the
instrument and quantifies what real E2 costs; it makes no claim about deployed
guardrails. Read the finding first:
[`docs/research/E2_DRYRUN_FINDING.md`](../../docs/research/E2_DRYRUN_FINDING.md).

```bash
PYTHONPATH=src python experiments/e2_dryrun/run_dryrun.py
python scripts/validate_e2_observations.py experiments/e2_dryrun/results/observations.jsonl
```

- `corpus.jsonl` — frozen authored item set (22 harmful, 22 benign), reference
  labels declared before any outcome.
- `run_dryrun.py` — the harness: frozen configs and reduction `h`, calibration on
  the benign subset, E2-conforming row emission, the excess-joint-failure estimand
  `Δ = p11 − pA·pB`, and the contract's pre-registered negative controls.
- `results/observations.jsonl` — a reference snapshot that passes the E2 validator.
- `results/report.json` — Δ, kernel bounds, sample complexity, and control outcomes.

The keyword and regex mechanisms are deterministic; the semantic mechanism depends
on the installed TF-IDF backend, so exact Δ can shift across SciPy/scikit-learn
versions — itself a small illustration of the Layer D version-identity concern the
real study must record. The committed snapshot is static and remains conformant
regardless; the test checks environment-robust invariants, not byte-equality of a
regenerated semantic column.
