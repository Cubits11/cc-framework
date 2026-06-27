# Paper Source Status

The current canonical paper-core narrative is
`docs/research/PAPER_CORE.md`, supported by the deterministic artifacts in
`artifacts/paper`.

`paper/main.tex` is a historical/in-progress LaTeX skeleton. It currently
references section files that are not present in this repository snapshot, so it
should not be treated as the current buildable manuscript for v0.3-rc1.

For the release-candidate artifact chain, use:

```bash
make reproduce-paper
make verify-paper-artifacts
```
