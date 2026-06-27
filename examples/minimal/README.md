# Minimal Bounds Example

This example is the smallest reviewer-facing atom-LP run in the repository.
It uses the kernel convention that `Z_i=1` denotes guardrail failure or unsafe
pass, declares exact singleton failure marginals, and identifies sharp bounds
for an AND composition query.

Run from the repository root:

```bash
PYTHONPATH=src .venv/bin/python examples/minimal/run_bounds.py
```

The JSON output includes the identified lower and upper bounds, FH width,
FH position for the supplied observed value, a product-coupling baseline,
lower/upper independence regret, and lower/upper endpoint witness checks.
