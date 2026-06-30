# CC Framework Repo Health Baseline

## Purpose

This document records the current repository health baseline for CC Framework after local documentation, typing, dependency, and test-collection checks.

The purpose is not to fix every issue in one pull request. The purpose is to classify the repo into clear remediation tracks so future PRs stay small, reviewable, and trustworthy.

## Current high-level finding

The repository is not failing because of one isolated bug. It currently has three overlapping classes of repository debt:

1. documentation infrastructure gaps,
2. runtime and test dependency packaging gaps,
3. staged typing and API-contract drift across core, cartographer, analysis, experiments, and legacy protocol modules.

These should remain separate cleanup tracks.

## Documentation infrastructure

The documentation workflow expected a MkDocs configuration. PR #58 restored the missing MkDocs configuration, removed strict-mode broken links to missing generated Week 5 figures, and added `site/` to `.gitignore`.

That PR established the documentation rule that committed docs must build strictly without depending on missing generated research artifacts.

Remaining documentation work should focus on improving architecture and governance docs without committing generated outputs such as `site/`, `results/`, `runs/`, `checkpoints/`, or regenerated paper artifacts.

## Runtime dependency packaging

Initial pytest collection failed before source tests could run because several packages used by tests and modules were not installed in the local `.[dev,docs]` environment.

Observed missing runtime dependencies included:

- pandas
- scipy
- matplotlib

Additional runtime/type dependencies needed for fuller local coverage included:

- scikit-learn
- statsmodels
- jsonschema
- prometheus-client
- pandas-stubs
- types-jsonschema

Current packaging already exposes several optional groups, including `test`,
`stats`, `viz`, `data`, `ml`, `docs`, `enterprise`, and `security`. Release
evidence should choose a lane from [Validation Matrix](../validation_matrix.md)
instead of implying that all optional surfaces were exercised by one command.

Useful install targets:

```bash
python -m pip install -e ".[dev,docs]"
python -m pip install -e ".[enterprise,test]"
python -m pip install -e ".[security]"
```

Optional dependency changes should still happen in dedicated packaging PRs, not
be mixed into typing or architecture cleanup.

## Typing and API-contract drift

The lint/type/test matrix currently surfaces broader typing and API-contract drift. These failures should not be collapsed into a single rescue PR.

The main areas to classify before fixing are:

- strict-kernel files versus non-kernel files,
- numeric policy consistency,
- serialization policy consistency,
- `J` naming conventions,
- CC denominator conventions,
- bounds return shapes,
- audit determinism requirements.

A future strict-kernel contract document should define these policies before large typing changes are attempted.

## Recommended remediation tracks

The staged cleanup should proceed in small PRs:

1. documentation infrastructure and strict docs build restoration,
2. repository health baseline documentation,
3. optional dependency group cleanup,
4. pytest collection restoration,
5. strict-kernel contract documentation,
6. focused typing repairs by subsystem,
7. source-level refactors only after contracts are explicit.

## Validation lanes

The active validation split is:

| Lane | Track | Primary command evidence |
| --- | --- | --- |
| Paper Core | Paper Core v0.3 | `make test-kernel`, `make test-release`, `make reproduce-paper`, `make verify-paper-artifacts`, `make paper-smoke` |
| Full Python | Paper Core v0.3 plus broader regression coverage | `PYTHONPATH=src .venv/bin/pytest -q` |
| Enterprise Reference | Enterprise Reference v0.1 | `PYTHONPATH=src .venv/bin/pytest tests/integration/test_enterprise_aws_emulation.py -q` with `.[enterprise,test]` |
| Dashboard | Enterprise Reference v0.1 application surface | `npm run build` in `apps/dashboard` plus `PYTHONPATH=src .venv/bin/pytest tests/e2e/test_enterprise_smoke.py -q` |
| Docs | Shared | `make docs` |
| Security | Shared package/source hygiene | `make security` |
| Optional Vendor | Optional adapter/vendor surfaces | Targeted adapter or performance tests with their vendor packages and gates installed |

Enterprise Reference evidence checks preserve evidence integrity. They do not
certify deployment safety, compliance, model correctness, policy correctness,
or operational readiness.

## Non-goals for this baseline

This document does not fix:

- mypy baseline errors,
- pytest collection errors,
- runtime dependency extras,
- source-code typing issues,
- generated research artifacts,
- paper figures or regenerated PDFs.

Those belong in separate follow-up PRs.
