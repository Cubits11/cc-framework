# Paper Source Status

`paper/main.tex` is the Paper 1 manuscript source for "Sharp Composition Bounds
for AI Guardrails Under Unknown Dependence." It is aligned with
`docs/research/PAPER_CORE.md` and the deterministic artifacts in
`artifacts/paper`.

For source and artifact checks, use:

```bash
make paper-smoke
```

For the deterministic artifact chain, use:

```bash
make reproduce-paper
make verify-paper-artifacts
```

`paper/draft.md` is retained only as a historical sketch of the earlier
two-setting scalar-metric framing. It is not Paper 1 source.
