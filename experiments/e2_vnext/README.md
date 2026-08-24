# E2-VNEXT — activation-gated external detector-miss study

**Status: NOT ACTIVATED.** The pre-loader gate result is intentionally the
only result under this directory. No model artifact was retrieved, no loader
was imported, no source text was parsed locally, and no candidate made a
prediction.

This directory exists to make the experiment's stopping conditions executable:

- [`FROZEN_DESIGN.json`](FROZEN_DESIGN.json) fixes the ontology, paired-Delta
  resolution target, order of gates, threshold-calibration rule, and the
  distinction between a detector-miss gate and application risk.
- [`power_design.py`](power_design.py) derives the required **2,025 complete
  paired attack items** from a two-sided 5% test with 95% power at
  `|Delta| = 0.020`; it is not inherited from E2-FIELD-TEST-001.
- [`CANDIDATE_SCREEN.json`](CANDIDATE_SCREEN.json) is an outcome-blind screen
  of public documentation only. It is not model output.
- [`preflight.py`](preflight.py) cannot download a model, open a dataset, or
  take a prompt. It writes the stop record only after evaluating the frozen
  gates.

The design uses a source-declared **agentic/indirect injection** target. It
does not relabel it as a direct-prompt or deployment population merely to make
it resemble an earlier protocol. The candidate screen fails strict
out-of-sample training-provenance eligibility, and no signed benchmark decision
cap exists. Those failures are terminal for this study identifier.

Run the non-loading preflight check:

```bash
python experiments/e2_vnext/preflight.py \
  --design experiments/e2_vnext/FROZEN_DESIGN.json \
  --candidate-screen experiments/e2_vnext/CANDIDATE_SCREEN.json \
  --decision-attestation experiments/e2_vnext/decision_attestation.json \
  --out experiments/e2_vnext/results/E2_VNEXT_PRELOADER_GATE_RECORD.json \
  --observed-at-utc 2026-08-24T00:00:00Z
```

The expected result is `NOT_ACTIVATED`; passing `--require-pass` makes that
expected stop condition exit nonzero for automation.
