# v0.3-rc1 Release Candidate Checklist

This checklist is for the paper-core artifact chain only. It does not expand the
release scope beyond the finite-atom kernel, canonical metrics, documentation
spine, and deterministic paper artifacts.

## Required validation commands

Run these commands from the repository root:

```bash
git status --short
make test-kernel
make test-release
rm -rf artifacts/paper && make reproduce-paper && make verify-paper-artifacts
make verify-paper-artifacts
env PYTHONPATH=src .venv/bin/pytest -q
env PYTHONPATH=src .venv/bin/mypy src/cc/kernel --strict
.venv/bin/ruff check src/cc/kernel tests/unit/kernel tests/integration scripts
.venv/bin/mkdocs build --strict --site-dir /tmp/cc-framework-mkdocs-site
git status --short
```

Expected behavior:

- The initial `git status --short` should show only intentional local work.
- `make test-kernel` must pass.
- `make test-release` must pass.
- `make reproduce-paper` must create a fresh `artifacts/paper` tree.
- Both `make verify-paper-artifacts` runs must pass, including the second run
  without regeneration.
- Full pytest must not create tracked diffs or new checkpoint artifacts.
- The final `git status --short` should show only intentional source,
  documentation, or regenerated paper-artifact changes.

## Optional dependency skips

- Optional notebooks, dashboard, cloud, and adapter paths are outside this
  release candidate.
- Pandoc-dependent PDF memo generation is optional unless a release note
  explicitly promotes it.
- The release is not blocked by unavailable optional plotting backends when the
  required validation commands above pass.

## Artifact verification procedure

1. Verify the committed artifact state before regenerating:

   ```bash
   make verify-paper-artifacts
   ```

2. Regenerate the paper artifacts from scratch:

   ```bash
   rm -rf artifacts/paper
   make reproduce-paper
   ```

3. Verify the regenerated artifacts:

   ```bash
   make verify-paper-artifacts
   make verify-paper-artifacts
   ```

The artifact directory is intended to be tracked for v0.3-rc1. Its manifest,
schema versions, hashes, witness distributions, tables, figures, and environment
metadata must be internally consistent.

## Fresh-venv validation notes

For a fresh environment:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip wheel setuptools
.venv/bin/pip install -e '.[dev,docs]'
```

Then run the required validation commands above. Do not treat missing optional
notebook, dashboard, cloud, or adapter dependencies as release blockers for this
paper-core RC.

## Non-claims

This release candidate does not claim:

- deployment safety,
- certification of deployed models,
- causal inference without causal assumptions,
- dataset representativeness,
- production readiness,
- enterprise readiness,
- a dashboard, cloud, or adapter release,
- a completed LaTeX manuscript.

The canonical paper-core narrative is `docs/research/PAPER_CORE.md`; generated
kernel artifacts live in `artifacts/paper`.

## Release blockers

Block v0.3-rc1 if any of the following are true:

- `artifacts/paper` fails verification before or after regeneration.
- Ordinary pytest creates tracked diffs or new checkpoint artifacts.
- The README describes `reproduce-paper` as merely planned.
- Public research docs or experiment READMEs make deployment, certification, or
  production-ready claims.
- `paper/main.tex` is presented as current while it references missing or stale
  section files.
- Mathematical behavior changes without a documented failing test and review.
