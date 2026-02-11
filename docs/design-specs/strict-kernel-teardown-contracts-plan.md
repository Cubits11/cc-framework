# CC-Framework Strict Kernel Teardown + Contracts Plan

## 1) Repo Reality Map (Verified)

### High-level tree (verified)
Top-level code + ops areas currently include:
- `src/cc/` (library code), with subpackages: `core`, `cartographer`, `analysis`, `exp`, `adapters`, `guardrails`, `io`, `utils`, `_legacy`, and `cli`.
- `tests/` (unit/integration/regression/experiments/performance/e2e).
- `experiments/` (standalone experiment runners including `correlation_cliff/`).
- `docs/`, `theory/`, `notebooks/`, `paper/`, `evaluation/`, `scripts/`, and deployment assets.

### Packaging identity
- Project name: `cc-framework`; version: `0.2.0`; Python `>=3.9`.
- Core dependencies are relatively small (`numpy`, `pydantic`, `pyyaml`, `jsonlines`, `cryptography`, `blake3`) with heavier stacks under extras (`stats`, `viz`, `ml`, `docs`, etc.).
- No active `project.scripts` entrypoint is currently enabled (commented out in `pyproject.toml`).
- Package discovery is src-layout (`src`), includes `cc*`, excludes `tests/benchmarks/experiments`.

### Source layout and public import surface
- Top-level package exports modules via `src/cc/__init__.py` and sets `__all__` to package-level namespaces (`adapters`, `analysis`, `cartographer`, `core`, `exp`, `guardrails`, `io`, `utils`).
- `src/cc/core/` exists but does **not** currently expose a canonical stable re-export boundary file (no `src/cc/core/__init__.py` and no dedicated `api_surface.py`).

### Test system and coverage areas
- Test framework is `pytest` (configured in `pyproject.toml`), including strict markers and deprecation policy.
- Property-based testing exists via Hypothesis (currently concentrated mostly in model/hash/parsing tests, not in a dedicated kernel invariants suite).
- Tests are broad: unit/integration/regression/experiments/performance and several week-based validation suites.

### Docs system/build
- CI has a docs workflow that installs docs extras and runs `make docs`.
- `Makefile` docs target uses `mkdocs build --strict`; serve target uses `mkdocs serve`.
- Repo references MkDocs, but no root `mkdocs.yml` is currently present in this checkout (workflow paths still include it).

### Experiments pipeline and determinism
- `experiments/run.py` validates config, records `git_sha`, config hash, dataset hash, and writes a run manifest.
- `src/cc/exp/run_two_world.py` sets deterministic controls (`PYTHONHASHSEED`, seeded Python/NumPy RNG, BLAS thread caps) and computes two-world metrics.
- `src/cc/io/storage.py` uses content-addressed/sharded path layout and deterministic hashing for datasets/files/directories.

### Existing audit/export artifacts
- Hash-chained JSONL audit support exists in `src/cc/cartographer/audit.py` (`append_jsonl`, `verify_chain`, `tail_sha`, etc.).
- Manifests are produced (`manifest.json`) and there are docs for deterministic storage layout.
- Audit data is currently spread across JSONL chain logs + manifests + generated outputs; there is no single formalized `audit_packet/` schema v1 contract yet.

### Existing theorem/assumption labeling
- There are governance templates and audit docs that mention assumptions and failure modes.
- A machine-readable kernel assumption registry with stable IDs and schema validation is not currently present as a first-class module.

### Explicit answers requested
- **Current public API:** de facto is module-level package import surface from `cc.__init__`, plus operational CLIs in `cc.cartographer.cli` and experiment scripts; not a narrow signed kernel API.
- **Core math primitives location:** split between `src/cc/core/metrics.py`, `src/cc/cartographer/bounds.py`/`stats.py`, and `experiments/correlation_cliff/theory_core.py`.
- **Dependence path / FH / envelope computations:** strongest/most explicit implementation currently sits in `experiments/correlation_cliff/theory_core.py` (FH bounds, `p11` paths, copulas, two-world bounds).
- **Where CC computed + edge policy:** CC-like quantities are computed in multiple places (`src/cc/core/metrics.py`, cartographer stats paths, and theory-core utilities) with differing normalization/degeneracy behavior and no single signed-off edge-case policy document.

---

## 2) Vision Mismatch (Strict Kernel vs Current)

### A) Blockers (must change)
1. **No signed kernel API boundary file with CI enforcement.**
2. **Math semantics are split/duplicated across `src/cc/core`, cartographer, and `experiments/correlation_cliff`.**
3. **No machine-readable assumption registry wired into computations.**
4. **No formal audit packet v1 schema with deterministic bundle contract (`manifest/results/report/figures`).**
5. **No API snapshot gate + changelog/migration enforcement for contract changes.**
6. **No explicit class-conditional-by-default kernel contract (Y=1 and Y=0 separated at API level).**

### B) Non-blocking divergences (can defer)
1. Existing CLI/experiment orchestration can stay while kernel extraction proceeds.
2. Existing docs/audit memos can be retained and back-linked into new contracts.
3. Existing hash-chain logging format can be adapted instead of replaced.

### C) Already aligned (keep)
1. Determinism intent exists (seed handling + thread capping).
2. Minimal core dependency direction already present.
3. Rich mathematical groundwork (FH bounds, copula modes, strict validation behavior) already exists in theory-core implementation.
4. CI already runs lint/type/tests and docs build hooks exist.

---

## 3) Strict Kernel Definition (Contracts + Invariants)

### 3.1 Contract artifacts and locations

#### API Contract
- `docs/contracts/api.md` (human contract, SemVer commitments, error policy).
- `src/cc/core/api_surface.py` (single canonical re-export surface).
- `src/cc/core/schemas.py` (typed request/response models).

#### Invariant Contract
- `docs/contracts/invariants.md` (formal properties + domain scope).
- `tests/property/test_core_invariants.py` (property checks).
- `tests/unit/core/test_kernel_edge_cases.py` (explicit corner cases).

#### Assumption Contract
- `docs/contracts/assumptions.md` (human narrative).
- `src/cc/core/assumptions_schema.py` (schema models).
- `src/cc/core/assumptions_registry.yaml` (machine-readable canonical list).

#### Repro Contract
- `docs/contracts/reproducibility.md` (seed/env policy).
- `src/cc/core/repro.py` (deterministic context helpers).

#### Audit Contract
- `docs/contracts/audit_packet_v1.md`.
- `src/cc/core/audit_packet.py` (pure deterministic builder + validators).
- `src/cc/core/audit_packet_schema.py`.

### 3.2 Operational “signed-off” definition
A contract is signed-off only when all are true:
1. Versioned contract file exists (docs + code-level surface/schema).
2. CI gate fails on unauthorized contract drift.
3. Contract-changing PR contains changelog entry and migration note (if breaking).

### Proposed CI checks (exact)
- `tests/contracts/test_api_snapshot.py`: hash exported symbols + signatures from `cc.core.api_surface`.
- `tests/contracts/test_schema_validation.py`: validate sample payloads for assumptions/audit packet.
- `tests/contracts/test_changelog_required.py`: fail if snapshot changes without `CHANGELOG.md` update.

---

## 4) Step-by-step Execution Plan (Steps 1–6)

### Step 1 — Freeze kernel boundary (cc-core contract)
**Exists now:** partial primitives in `src/cc/core/metrics.py` + advanced dependence math in `experiments/correlation_cliff/theory_core.py`.

**Create/change:**
- Create `src/cc/core/api_surface.py` with stable signatures and strict typed contracts.
- Create `docs/contracts/api.md` 1-page canonical spec.
- Add deprecation wrappers for legacy function paths that now resolve to api-surface.

**File-level plan:**
- Add: `src/cc/core/api_surface.py`, `src/cc/core/schemas.py`, `docs/contracts/api.md`.
- Update: `src/cc/__init__.py` (optionally expose `core.api_surface` via stable alias), legacy modules with `DeprecationWarning`.

**Done criteria:**
- API snapshot test green.
- Deprecated imports emit warning + preserve behavior.
- No math duplication for signed functions.

### Step 2 — Operators as types (OR / AND / Sequential)
**Exists now:** semantics are function-based and distributed.

**Create/change:**
- `src/cc/core/operators.py` dataclasses: `AndOperator`, `OrOperator`, `SequentialOperator`.
- Support modes: `observed_joint`, `fh_worst`, `fh_best`, `copula(param)`.

**Done criteria:**
- Operators accept either raw paired outcomes or marginals + dependence model.
- Class-conditional outputs returned separately by contract.

### Step 3 — Machine-readable Assumption Registry
**Exists now:** assumptions are mostly prose templates/docs.

**Create/change:**
- `src/cc/core/assumptions_registry.yaml` + `assumptions_schema.py` + loader/validator.
- Thread assumption IDs through kernel outputs and audit exports.

**Done criteria:**
- Registry validates in CI.
- Every verdict-producing call emits assumption IDs used.

### Step 4 — Audit Packet v1 format
**Exists now:** manifests + hash-chain logs + report generation pieces.

**Create/change:**
- Deterministic `audit_packet/` bundle contract with `manifest.json`, `results.json`, `report.md`, `figures/`.
- Add builder in `src/cc/core/audit_packet.py` and schema tests.

**Done criteria:**
- Golden packet fixture reproducible byte-for-byte (except explicitly allowed timestamp fields if excluded).
- CI verifies schema + deterministic regen.

### Step 5 — Upgrade GCE into thin client of cc-core
**Exists now:** not directly verifiable in this repo as separate service/repo.

**Create/change:**
- Define kernel entrypoints for external clients and publish version pinning policy.
- Enforce no duplicated math in clients (client only calls kernel APIs).

**Done criteria:**
- Client contract doc references tagged `cc-core` release.
- Integration tests call kernel API only.

### Step 6 — Build cc-academy (separate repo)
**Exists now:** notebooks/docs exist in current repo but not a separated academy package.

**Create/change:**
- Specify educational-facing stable read-only APIs, notebook examples, and CI recipe for headless execution.

**Done criteria:**
- Notebook smoke tests deterministic in CI.
- academy repo consumes pinned core package.

---

## 5) API Spec v0 (1 page)

```python
# src/cc/core/api_surface.py (proposed)
from typing import Literal
from .schemas import (
    ClassConditionalRates,
    OperatorRequest,
    OperatorResult,
    CCBoundsRequest,
    CCBoundsResult,
)

def evaluate_operator(req: OperatorRequest) -> OperatorResult: ...
def compute_cc_bounds(req: CCBoundsRequest) -> CCBoundsResult: ...
def validate_assumptions(ids: list[str]) -> None: ...
def build_audit_packet(result: CCBoundsResult, *, out_dir: str) -> str: ...
```

### Core schema sketch (proposed)
- `ClassConditionalRates`: `{tpr: float, fpr: float, support_pos: int, support_neg: int}`
- `OperatorRequest`:
  - `operator: Literal["AND","OR","SEQUENTIAL"]`
  - `mode: Literal["observed_joint","fh_worst","fh_best","copula"]`
  - `y1` and `y0` sections required (class-conditional separation)
  - optional dependence params (`lambda`, `copula_family`, `copula_param`)
  - `assumption_ids: list[str]`
- `OperatorResult`:
  - class-conditional composed rates
  - FH interval metadata
  - invariant checks and pass/fail details
  - assumption IDs echoed

### Numeric-only meta rule (proposed)
- Allowed scalar meta value types: finite `int|float|bool` only.
- Lists/dicts in machine fields must recursively resolve to numeric/bool scalars.
- Human-readable text goes only in `report.md`, never in `results.json` machine payload.

### Error policy (proposed)
- **Kernel strict mode default:** raise typed exceptions (`InputValidationError`, `FeasibilityError`, `InvariantViolationError`).
- Optional non-strict wrapper may clip/warn, but strict kernel API itself defaults to raise.

### Stability promise (proposed)
- SemVer-stable: `cc.core.api_surface` signatures + schemas in `schemas.py` + audit packet v1 schema.
- Experimental (no stability promise): exploratory helpers in `cc.analysis`, `experiments/`, notebook utilities.

---

## 6) Invariants & Property Tests (Executable)

### Required invariants
1. FH feasibility respected: returned `p11` in `[L,U]`.
2. OR monotonicity under fixed marginals assumptions.
3. AND monotonicity under fixed marginals assumptions.
4. Corner optimality: extrema achieved at FH bounds.
5. CC finite/sentinel contract: no unhandled NaN/Inf.
6. Class-conditional separation: Y=1 and Y=0 paths cannot be silently merged.
7. Deterministic replay: same inputs + seed => identical outputs.
8. Numeric-only meta policy always enforced.
9. Audit packet schemas valid.
10. Assumption IDs must exist in registry for strict runs.

### Proposed property tests (10)
1. `test_p11_always_within_fh_bounds`
2. `test_or_extrema_match_fh_endpoints`
3. `test_and_extrema_match_fh_endpoints`
4. `test_cc_bounds_nonnegative_and_ordered`
5. `test_cc_degeneracy_jbest_zero_returns_sentinel_or_zero`
6. `test_operator_output_stable_under_seed`
7. `test_class_conditional_required_fields`
8. `test_numeric_meta_rejects_strings`
9. `test_assumption_ids_roundtrip`
10. `test_audit_packet_results_all_numeric`

### Proposed edge-case unit tests (10)
1. `pA=0,pB=0`
2. `pA=1,pB=1`
3. `pA=1,pB=0`
4. `pA≈0,pB≈1` with epsilon clips
5. invalid `p11` below `L`
6. invalid `p11` above `U`
7. empty class-conditional sample arrays
8. copula parameter out-of-domain
9. `J_best=0` normalization branch
10. deterministic packet regeneration byte equality

---

## 7) Assumption Registry Spec + Initial Entries

### Schema (proposed)
```yaml
- id: ASM-001
  name: binary_outcomes
  scope: [operator_eval, cc_bounds]
  description: Outcomes are Bernoulli in {0,1}
  falsifiability:
    check: "all outcomes in dataset are exactly 0 or 1"
    severity: error
  implications:
    if_violated: "kernel metrics invalid; abort strict run"
```

### Initial 12 entries (proposed)
1. `ASM-001` binary outcomes.
2. `ASM-002` class labels are well-defined for Y=1/Y=0 splits.
3. `ASM-003` sample representativeness to deployment domain.
4. `ASM-004` stationarity within each world window.
5. `ASM-005` threshold realizability for component rails.
6. `ASM-006` independence claim only when explicitly selected mode says so.
7. `ASM-007` FH-feasible marginals/joint consistency.
8. `ASM-008` copula family adequacy when copula mode is used.
9. `ASM-009` missingness mechanism documented (MCAR/MAR/MNAR).
10. `ASM-010` label noise bounded and monitored.
11. `ASM-011` deterministic seed policy honored.
12. `ASM-012` calibration window validity for selected operating points.

---

## 8) Audit Packet v1 Spec

### `manifest.json` schema (proposed; required fields)
- `packet_version` (string, e.g., `"1.0"`)
- `git_commit` (sha)
- `python_version`
- `os`
- `deps_hash`
- `random_seed`
- `dataset_fingerprint`
- `run_id`
- `created_at_utc`
- `assumption_ids`
- `input_hash`

### `results.json` schema (numeric-only machine payload)
- `cc_point`
- `cc_lower`
- `cc_upper`
- `jc_point`
- `jc_lower`
- `jc_upper`
- `mode_code` (enum encoded as integer for strict numeric-only payload)
- `operator_code`
- `tpr_y1`, `fpr_y0`, supports, and other finite numeric diagnostics

### `report.md` template
Sections:
1. Run context
2. Assumptions engaged
3. Operator/mode used
4. Primary outcomes (CC/JC + bounds)
5. Violations or warnings
6. Repro instructions

### `figures/` naming convention
- `figures/01_roc_envelope.png`
- `figures/02_cc_path.png`
- `figures/03_sensitivity.png`

---

## 9) Refactor Plan (file moves, names, deprecations)

### Proposed module layout under `src/cc/core`
- `api_surface.py` (stable exports only)
- `schemas.py`
- `operators.py`
- `dependence.py` (FH + copulas)
- `cc_metrics.py`
- `assumptions_schema.py`
- `assumptions_registry.yaml`
- `audit_packet.py`
- `repro.py`
- `errors.py`

### Deprecations
- Keep wrappers in current `metrics.py`/cartographer paths for 1 minor release.
- Emit `DeprecationWarning` with migration target.

### Name normalization
- Keep distribution package as `cc-framework`, but contract docs should refer to strict kernel as `cc-core` sub-surface.

### Duplication elimination
- Promote validated math from `experiments/correlation_cliff/theory_core.py` into `src/cc/core/*`.
- `experiments/` and `cartographer` become clients of kernel API.

---

## 10) CI Gates & Release Discipline

### Required CI gates (proposed)
1. Unit tests + property tests.
2. Lint + format + type checks.
3. Golden artifact determinism check (`audit_packet` fixtures regenerate identically).
4. API snapshot check + changelog requirement for API drift.
5. Schema validation for assumptions + audit packet.
6. Optional docs build check retained.

### SemVer policy (proposed)
- **Major:** breaking API signature/schema changes in signed surface.
- **Minor:** additive stable API/schema fields, new operators/modes, backward-compatible extensions.
- **Patch:** bug fixes, perf improvements, docs/tooling.

### Audit schema evolution
- Additive optional fields => minor.
- Required-field changes/removals => major with migration note.

---

## 11) Open Questions (truly unanswerable from repo)

1. Should strict kernel default CC formulation be ratio-based (`JC/J_best`) only, or also expose log-ratio as canonical?
2. For strict mode, should edge invalidity default to `raise` universally, or allow policy-selectable `clip` in stable API?
3. Should numeric-only machine payload encode enums as integers (strictest) or allow bounded strings for readability?
4. What is the canonical dependency boundary contract for external GCE (tag-only pin vs compatible range)?

---

## Evidence index (key files inspected)
- `pyproject.toml`
- `src/cc/__init__.py`
- `src/cc/core/metrics.py`
- `src/cc/exp/run_two_world.py`
- `experiments/run.py`
- `src/cc/io/storage.py`
- `src/cc/cartographer/audit.py`
- `src/cc/cartographer/cli.py`
- `experiments/correlation_cliff/theory_core.py`
- `.github/workflows/ci.yml`
- `.github/workflows/docs.yml`
- `Makefile`
- `docs/architecture/storage_layout.md`
- `docs/ADAPTER_TESTING.md`
