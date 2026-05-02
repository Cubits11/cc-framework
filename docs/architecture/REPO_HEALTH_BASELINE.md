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

This suggests that the repository needs clearer optional dependency groups such as `test`, `type`, `stats`, `viz`, and possibly `full`.

Recommended future install target:

```bash
python -m pip install -e ".[dev,docs,test,type]"
```

The exact extras should be introduced in a dedicated packaging PR, not mixed into documentation or typing cleanup.

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

## Non-goals for this baseline

This document does not fix:

- mypy baseline errors,
- pytest collection errors,
- runtime dependency extras,
- source-code typing issues,
- generated research artifacts,
- paper figures or regenerated PDFs.

Those belong in separate follow-up PRs.
