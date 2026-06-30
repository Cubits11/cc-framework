#!/usr/bin/env python
"""Verify paper reproduction artifacts and LP endpoint witnesses."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from jsonschema import Draft202012Validator, ValidationError

from cc.evals.dependence_benchmark import verify_benchmark_summary
from cc.kernel.frechet_classes import classical_frechet_bounds
from cc.kernel.metrics import (
    cc_gain,
    cc_shift,
    fh_position,
    fh_width,
    independence_regret,
    independent_event_probability,
)
from cc.kernel.sample_complexity import (
    simultaneous_bernoulli_radius,
    simultaneous_sample_size,
)
from cc.kernel.sensitivity import (
    AssumptionSet,
    LinearConstraint,
    LinearQuery,
    enumerate_atoms,
    identified_region,
)

REQUIRED_FILES = (
    "table_1_classical_frechet_bounds.csv",
    "table_2_metric_examples.csv",
    "table_3_witness_verification.csv",
    "table_4_sample_complexity.csv",
    "table_5_runtime_scaling.csv",
    "figure_1_fh_interval.png",
    "figure_2_independence_regret.png",
    "figure_3_correlation_cliff_toy.png",
    "figure_4_runtime_scaling.png",
    "benchmark_example_summary.json",
    "minimal_bounds.json",
    "minimal_witnesses.json",
    "minimal_bundle.json",
    "environment.json",
    "manifest.json",
)
HASHED_FILES = tuple(name for name in REQUIRED_FILES if name != "manifest.json")
DEFAULT_ARTIFACT_DIR = Path("artifacts/paper")
DEFAULT_TOL = 1.0e-8
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "generation_command",
        "package_version",
        "required_files",
        "files",
        "manifest_payload_sha256",
    ],
    "properties": {
        "schema_version": {"const": "cc.paper.manifest.v1"},
        "generation_command": {"type": "string", "minLength": 1},
        "package_version": {"type": ["string", "null"]},
        "required_files": {"type": "array", "items": {"type": "string"}},
        "files": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["filename", "sha256", "bytes"],
                "properties": {
                    "filename": {"type": "string"},
                    "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    "bytes": {"type": "integer", "minimum": 0},
                },
            },
        },
        "manifest_payload_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
}

BOUNDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "failure_event_convention", "cases"],
    "properties": {
        "schema_version": {"const": "cc.paper.bounds.v1"},
        "failure_event_convention": {"type": "string", "minLength": 1},
        "cases": {"type": "array", "minItems": 1},
    },
}

WITNESSES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "atom_order", "tolerance", "cases"],
    "properties": {
        "schema_version": {"const": "cc.paper.witnesses.v1"},
        "atom_order": {"const": "little_endian"},
        "tolerance": {"type": "number", "exclusiveMinimum": 0},
        "cases": {"type": "array", "minItems": 1},
    },
}

BUNDLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "artifact_files",
        "bounds_file",
        "witnesses_file",
        "environment_file",
        "cases",
    ],
    "properties": {
        "schema_version": {"const": "cc.paper.bundle.v1"},
        "artifact_files": {"type": "array", "items": {"type": "string"}},
        "bounds_file": {"const": "minimal_bounds.json"},
        "witnesses_file": {"const": "minimal_witnesses.json"},
        "environment_file": {"const": "environment.json"},
        "cases": {"type": "array", "minItems": 1},
    },
}

ENVIRONMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_version",
        "python",
        "platform",
        "package_version",
        "dependency_versions",
        "timestamp_policy",
        "generation_timestamp",
        "generation_command",
    ],
    "properties": {
        "schema_version": {"const": "cc.paper.environment.v1"},
        "python": {"type": "object"},
        "platform": {"type": "object"},
        "package_version": {"type": ["string", "null"]},
        "dependency_versions": {"type": "object"},
        "timestamp_policy": {"type": "string", "minLength": 1},
        "generation_timestamp": {"type": "string", "minLength": 1},
        "generation_command": {"type": "string", "minLength": 1},
    },
}


class ArtifactVerificationError(RuntimeError):
    """Raised when paper artifact verification fails."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        "--dir",
        dest="artifact_dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Artifact directory to verify.",
    )
    parser.add_argument("--tol", type=float, default=DEFAULT_TOL, help="Numerical tolerance.")
    args = parser.parse_args()

    try:
        verify_artifact_dir(args.artifact_dir, tol=args.tol)
    except ArtifactVerificationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Verified paper artifacts in {args.artifact_dir}")
    return 0


def verify_artifact_dir(artifact_dir: Path, *, tol: float = DEFAULT_TOL) -> None:
    if tol <= 0.0 or not np.isfinite(tol):
        raise ArtifactVerificationError("Verification tolerance must be positive and finite.")

    errors: list[str] = []
    errors.extend(_check_required_files(artifact_dir))
    if errors:
        raise ArtifactVerificationError("Paper artifact verification failed:\n- " + "\n- ".join(errors))

    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.exists():
        raise ArtifactVerificationError("manifest.json is missing.")

    manifest = _load_json(manifest_path)
    errors.extend(_validate_schema("manifest.json", manifest, MANIFEST_SCHEMA))
    if not errors:
        errors.extend(_verify_manifest(artifact_dir, manifest))

    loaded_json = _load_and_validate_json_artifacts(artifact_dir)
    for filename, payload, schema in loaded_json:
        errors.extend(_validate_schema(filename, payload, schema))

    try:
        bounds = _load_json(artifact_dir / "minimal_bounds.json")
        witnesses = _load_json(artifact_dir / "minimal_witnesses.json")
        bundle = _load_json(artifact_dir / "minimal_bundle.json")
        environment = _load_json(artifact_dir / "environment.json")
        benchmark_example = _load_json(artifact_dir / "benchmark_example_summary.json")
    except OSError as exc:
        errors.append(f"Unable to read JSON artifact: {exc}")
    else:
        errors.extend(_verify_environment(environment))
        errors.extend(_verify_bundle(bundle))
        errors.extend(_verify_bounds_against_witnesses(bounds, witnesses, tol=tol))
        errors.extend(_verify_bounds_metrics(bounds, tol=tol))
        errors.extend(_verify_witnesses(witnesses, tol=tol))
        errors.extend(_verify_benchmark_example(benchmark_example))

    errors.extend(_verify_classical_table(artifact_dir / "table_1_classical_frechet_bounds.csv", tol=tol))
    errors.extend(_verify_metric_table(artifact_dir / "table_2_metric_examples.csv", tol=tol))
    errors.extend(_verify_witness_table(artifact_dir / "table_3_witness_verification.csv", tol=tol))
    errors.extend(_verify_sample_complexity_table(artifact_dir / "table_4_sample_complexity.csv", tol=tol))
    errors.extend(_verify_runtime_scaling_table(artifact_dir / "table_5_runtime_scaling.csv"))
    errors.extend(_verify_png_artifacts(artifact_dir))

    if errors:
        raise ArtifactVerificationError("Paper artifact verification failed:\n- " + "\n- ".join(errors))


def _check_required_files(artifact_dir: Path) -> list[str]:
    errors: list[str] = []
    for filename in REQUIRED_FILES:
        path = artifact_dir / filename
        if not path.is_file():
            errors.append(f"Required artifact is missing: {filename}")
    return errors


def _verify_manifest(artifact_dir: Path, manifest: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    required = set(_string_list(manifest.get("required_files"), label="manifest.required_files"))
    expected = set(REQUIRED_FILES)
    if required != expected:
        errors.append(f"Manifest required files mismatch: expected {sorted(expected)}, got {sorted(required)}")

    manifest_hash = manifest.get("manifest_payload_sha256")
    expected_manifest_hash = _hash_manifest_payload(manifest)
    if manifest_hash != expected_manifest_hash:
        errors.append("manifest_payload_sha256 does not match canonical manifest payload.")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        return [*errors, "manifest.files must be a list."]

    file_entries: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            errors.append("manifest.files contains a non-object entry.")
            continue
        filename = entry.get("filename")
        if not isinstance(filename, str):
            errors.append("manifest.files entry is missing a string filename.")
            continue
        file_entries[filename] = entry

    missing_hashes = sorted(set(HASHED_FILES) - set(file_entries))
    if missing_hashes:
        errors.append(f"Manifest is missing hash entries for: {missing_hashes}")

    for filename, entry in sorted(file_entries.items()):
        path = artifact_dir / filename
        if not path.is_file():
            errors.append(f"Manifest hashes missing file: {filename}")
            continue
        recorded_hash = entry.get("sha256")
        actual_hash = _sha256_file(path)
        if recorded_hash != actual_hash:
            errors.append(f"Hash mismatch for {filename}: manifest {recorded_hash}, actual {actual_hash}")
        recorded_bytes = entry.get("bytes")
        actual_bytes = path.stat().st_size
        if recorded_bytes != actual_bytes:
            errors.append(f"Byte-size mismatch for {filename}: manifest {recorded_bytes}, actual {actual_bytes}")
    return errors


def _load_and_validate_json_artifacts(artifact_dir: Path) -> list[tuple[str, Any, Mapping[str, Any]]]:
    return [
        ("minimal_bounds.json", _load_json(artifact_dir / "minimal_bounds.json"), BOUNDS_SCHEMA),
        ("minimal_witnesses.json", _load_json(artifact_dir / "minimal_witnesses.json"), WITNESSES_SCHEMA),
        ("minimal_bundle.json", _load_json(artifact_dir / "minimal_bundle.json"), BUNDLE_SCHEMA),
        ("environment.json", _load_json(artifact_dir / "environment.json"), ENVIRONMENT_SCHEMA),
    ]


def _validate_schema(filename: str, payload: Any, schema: Mapping[str, Any]) -> list[str]:
    try:
        Draft202012Validator(schema).validate(payload)
    except ValidationError as exc:
        return [f"{filename} is not schema-valid: {exc.message}"]
    return []


def _verify_environment(environment: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    python_payload = environment.get("python")
    platform_payload = environment.get("platform")
    dependency_versions = environment.get("dependency_versions")
    if not isinstance(python_payload, Mapping) or "version" not in python_payload:
        errors.append("environment.json is missing python.version metadata.")
    if not isinstance(platform_payload, Mapping) or "system" not in platform_payload:
        errors.append("environment.json is missing platform.system metadata.")
    if not isinstance(dependency_versions, Mapping) or "numpy" not in dependency_versions:
        errors.append("environment.json is missing relevant dependency versions.")
    return errors


def _verify_bundle(bundle: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    artifact_files = set(_string_list(bundle.get("artifact_files"), label="bundle.artifact_files"))
    if artifact_files != set(REQUIRED_FILES):
        errors.append("minimal_bundle.json does not list exactly the required artifact files.")
    if bundle.get("bounds_file") != "minimal_bounds.json":
        errors.append("minimal_bundle.json points to the wrong bounds file.")
    if bundle.get("witnesses_file") != "minimal_witnesses.json":
        errors.append("minimal_bundle.json points to the wrong witnesses file.")
    return errors


def _verify_bounds_against_witnesses(
    bounds: Mapping[str, Any],
    witnesses: Mapping[str, Any],
    *,
    tol: float,
) -> list[str]:
    errors: list[str] = []
    bounds_cases = _cases_by_id(bounds.get("cases"), label="minimal_bounds.json")
    witness_cases = _cases_by_id(witnesses.get("cases"), label="minimal_witnesses.json")
    if set(bounds_cases) != set(witness_cases):
        return [f"Bounds and witness case ids differ: {sorted(bounds_cases)} vs {sorted(witness_cases)}"]
    for case_id, bounds_case in sorted(bounds_cases.items()):
        witness_case = witness_cases[case_id]
        for key in ("lower_bound", "upper_bound"):
            witness_key = "lower" if key == "lower_bound" else "upper"
            expected = float(bounds_case[key])
            actual = float(witness_case["bounds"][witness_key])
            if abs(expected - actual) > tol:
                errors.append(f"{case_id} {key} differs between bounds and witnesses.")
        for key in ("labels", "declared_marginals", "query", "assumptions_hash"):
            if bounds_case.get(key) != witness_case.get(key):
                errors.append(f"{case_id} {key} differs between bounds and witnesses.")
    return errors


def _verify_witnesses(witnesses: Mapping[str, Any], *, tol: float) -> list[str]:
    errors: list[str] = []
    cases = witnesses.get("cases")
    if not isinstance(cases, list):
        return ["minimal_witnesses.json cases must be a list."]
    for case in cases:
        if not isinstance(case, Mapping):
            errors.append("minimal_witnesses.json contains a non-object case.")
            continue
        errors.extend(_verify_witness_case(case, tol=tol))
    return errors


def _verify_witness_case(case: Mapping[str, Any], *, tol: float) -> list[str]:
    case_id = str(case.get("case_id", "<unknown>"))
    errors: list[str] = []
    try:
        labels = tuple(_string_list(case["labels"], label=f"{case_id}.labels"))
        n_atoms = 1 << len(labels)
        declared_marginals = _float_mapping(case["declared_marginals"], label=f"{case_id}.declared_marginals")
        assumptions_payload = _mapping(case["assumptions"], label=f"{case_id}.assumptions")
        query_payload = _mapping(case["query"], label=f"{case_id}.query")
        bounds = _mapping(case["bounds"], label=f"{case_id}.bounds")
        witnesses = _mapping(case["witnesses"], label=f"{case_id}.witnesses")
        query_coefficients = _float_vector(
            query_payload["coefficients"],
            expected_length=n_atoms,
            label=f"{case_id}.query.coefficients",
        )
        reported_lower = float(bounds["lower"])
        reported_upper = float(bounds["upper"])
    except (KeyError, TypeError, ValueError) as exc:
        return [f"{case_id} has malformed witness payload: {exc}"]

    errors.extend(_verify_assumptions_hash(case, labels, assumptions_payload))
    errors.extend(
        _verify_reported_endpoints_are_lp_optima(
            case_id,
            labels=labels,
            assumptions_payload=assumptions_payload,
            query_payload=query_payload,
            query_coefficients=query_coefficients,
            reported_lower=reported_lower,
            reported_upper=reported_upper,
            tol=tol,
        )
    )

    for endpoint in ("lower", "upper"):
        witness_payload = witnesses.get(endpoint)
        if not isinstance(witness_payload, Mapping):
            errors.append(f"{case_id}.{endpoint} witness is missing.")
            continue
        try:
            distribution = _float_vector(
                witness_payload["distribution"],
                expected_length=n_atoms,
                label=f"{case_id}.{endpoint}.distribution",
            )
            recorded_query_value = float(witness_payload["query_value"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{case_id}.{endpoint} witness is malformed: {exc}")
            continue
        errors.extend(
            _verify_distribution(
                case_id,
                endpoint,
                distribution,
                labels=labels,
                declared_marginals=declared_marginals,
                assumptions_payload=assumptions_payload,
                tol=tol,
            )
        )
        computed_query_value = float(query_coefficients @ distribution)
        if abs(computed_query_value - recorded_query_value) > tol:
            errors.append(
                f"{case_id}.{endpoint} recorded query value {recorded_query_value} "
                f"does not equal witness value {computed_query_value}."
            )
        reported_bound = reported_lower if endpoint == "lower" else reported_upper
        if abs(computed_query_value - reported_bound) > tol:
            errors.append(
                f"{case_id}.{endpoint} witness value {computed_query_value} "
                f"does not match reported bound {reported_bound}."
            )
    return errors


def _verify_reported_endpoints_are_lp_optima(
    case_id: str,
    *,
    labels: Sequence[str],
    assumptions_payload: Mapping[str, Any],
    query_payload: Mapping[str, Any],
    query_coefficients: np.ndarray[Any, Any],
    reported_lower: float,
    reported_upper: float,
    tol: float,
) -> list[str]:
    try:
        assumptions = _assumptions_from_payload(labels, assumptions_payload)
        query = LinearQuery.from_coefficients(
            labels,
            str(query_payload.get("name", f"{case_id}_query")),
            query_coefficients,
        )
        result = identified_region(query, assumptions, feasibility_tol=tol)
    except (TypeError, ValueError) as exc:
        return [f"{case_id} could not be re-solved from witness payload: {exc}"]

    errors: list[str] = []
    if abs(result.lower_bound - reported_lower) > tol:
        errors.append(
            f"{case_id} reported lower endpoint {reported_lower} "
            f"does not match recomputed LP optimum {result.lower_bound}."
        )
    if abs(result.upper_bound - reported_upper) > tol:
        errors.append(
            f"{case_id} reported upper endpoint {reported_upper} "
            f"does not match recomputed LP optimum {result.upper_bound}."
        )
    return errors


def _verify_distribution(
    case_id: str,
    endpoint: str,
    distribution: np.ndarray[Any, Any],
    *,
    labels: Sequence[str],
    declared_marginals: Mapping[str, float],
    assumptions_payload: Mapping[str, Any],
    tol: float,
) -> list[str]:
    errors: list[str] = []
    if not np.all(np.isfinite(distribution)):
        errors.append(f"{case_id}.{endpoint} distribution contains non-finite values.")
    min_probability = float(np.min(distribution))
    if min_probability < -tol:
        errors.append(f"{case_id}.{endpoint} distribution has negative mass {min_probability}.")
    probability_sum = float(np.sum(distribution))
    if abs(probability_sum - 1.0) > tol:
        errors.append(f"{case_id}.{endpoint} distribution sums to {probability_sum}, not 1.")
    errors.extend(
        _verify_declared_marginals(
            case_id,
            endpoint,
            distribution,
            labels=labels,
            declared_marginals=declared_marginals,
            tol=tol,
        )
    )
    errors.extend(
        _verify_constraints(
            case_id,
            endpoint,
            distribution,
            assumptions_payload=assumptions_payload,
            tol=tol,
        )
    )
    return errors


def _verify_declared_marginals(
    case_id: str,
    endpoint: str,
    distribution: np.ndarray[Any, Any],
    *,
    labels: Sequence[str],
    declared_marginals: Mapping[str, float],
    tol: float,
) -> list[str]:
    errors: list[str] = []
    atoms = enumerate_atoms(labels).astype(float)
    marginals = atoms.T @ distribution
    for index, label in enumerate(labels):
        expected = declared_marginals.get(label)
        if expected is None:
            errors.append(f"{case_id}.{endpoint} is missing declared marginal for {label}.")
            continue
        actual = float(marginals[index])
        if abs(actual - expected) > tol:
            errors.append(
                f"{case_id}.{endpoint} marginal {label} is {actual}, expected {expected}."
            )
    return errors


def _verify_constraints(
    case_id: str,
    endpoint: str,
    distribution: np.ndarray[Any, Any],
    *,
    assumptions_payload: Mapping[str, Any],
    tol: float,
) -> list[str]:
    errors: list[str] = []
    constraints = assumptions_payload.get("constraints")
    if not isinstance(constraints, list):
        return [f"{case_id}.{endpoint} assumptions.constraints must be a list."]
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            errors.append(f"{case_id}.{endpoint} contains a non-object constraint.")
            continue
        try:
            name = str(constraint["name"])
            sense = str(constraint["sense"])
            rhs = float(constraint["rhs"])
            coefficients = _float_vector(
                constraint["coefficients"],
                expected_length=distribution.size,
                label=f"{case_id}.{endpoint}.{name}.coefficients",
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"{case_id}.{endpoint} malformed constraint: {exc}")
            continue
        lhs = float(coefficients @ distribution)
        if sense == "==" and abs(lhs - rhs) > tol:
            errors.append(f"{case_id}.{endpoint} violates {name}: {lhs} != {rhs}.")
        elif sense == "<=" and lhs > rhs + tol:
            errors.append(f"{case_id}.{endpoint} violates {name}: {lhs} > {rhs}.")
        elif sense == ">=" and lhs + tol < rhs:
            errors.append(f"{case_id}.{endpoint} violates {name}: {lhs} < {rhs}.")
        elif sense not in {"==", "<=", ">="}:
            errors.append(f"{case_id}.{endpoint} has unknown constraint sense {sense!r}.")
    return errors


def _verify_assumptions_hash(
    case: Mapping[str, Any],
    labels: Sequence[str],
    assumptions_payload: Mapping[str, Any],
) -> list[str]:
    try:
        assumptions = _assumptions_from_payload(labels, assumptions_payload)
    except (TypeError, ValueError, KeyError) as exc:
        return [f"Unable to reconstruct assumptions for hash verification: {exc}"]
    expected = assumptions.stable_hash()
    actual = case.get("assumptions_hash")
    if actual != expected:
        return [f"{case.get('case_id', '<unknown>')} assumptions_hash mismatch."]
    return []


def _assumptions_from_payload(
    labels: Sequence[str],
    assumptions_payload: Mapping[str, Any],
) -> AssumptionSet:
    constraints_payload = assumptions_payload.get("constraints")
    if not isinstance(constraints_payload, list):
        raise TypeError("assumptions.constraints must be a list.")
    constraints = tuple(
        LinearConstraint.from_coefficients(
            labels,
            str(constraint["name"]),
            constraint["coefficients"],
            str(constraint["sense"]),
            float(constraint["rhs"]),
        )
        for constraint in constraints_payload
        if isinstance(constraint, Mapping)
    )
    metadata = _mapping(assumptions_payload.get("metadata", {}), label="assumptions.metadata")
    return AssumptionSet(tuple(labels), constraints=constraints, metadata=metadata)


def _verify_bounds_metrics(bounds: Mapping[str, Any], *, tol: float) -> list[str]:
    errors: list[str] = []
    cases = bounds.get("cases")
    if not isinstance(cases, list):
        return ["minimal_bounds.json cases must be a list."]
    for case in cases:
        if not isinstance(case, Mapping):
            errors.append("minimal_bounds.json contains a non-object case.")
            continue
        case_id = str(case.get("case_id", "<unknown>"))
        try:
            labels = tuple(_string_list(case["labels"], label=f"{case_id}.labels"))
            marginals = _float_mapping(
                case["declared_marginals"],
                label=f"{case_id}.declared_marginals",
            )
            query_payload = _mapping(case["query"], label=f"{case_id}.query")
            query = LinearQuery.from_coefficients(
                labels,
                str(query_payload.get("name", f"{case_id}_query")),
                _float_vector(
                    query_payload["coefficients"],
                    expected_length=1 << len(labels),
                    label=f"{case_id}.query.coefficients",
                ),
            )
            lower = float(case["lower_bound"])
            upper = float(case["upper_bound"])
            independent = independent_event_probability(marginals, query, labels=labels)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"Malformed minimal bounds case {case_id}: {exc}")
            continue

        errors.extend(
            _check_optional_metric(
                case,
                "fh_width",
                fh_width(lower, upper),
                case_id=case_id,
                tol=tol,
            )
        )
        errors.extend(
            _check_optional_metric(
                case,
                "independent_baseline",
                independent,
                case_id=case_id,
                tol=tol,
            )
        )
        errors.extend(
            _check_optional_metric(
                case,
                "product_baseline",
                independent,
                case_id=case_id,
                tol=tol,
            )
        )
        if "observed" in case:
            try:
                observed = float(case["observed"])
                position = fh_position(observed, lower, upper)
            except (TypeError, ValueError) as exc:
                errors.append(f"{case_id}.observed is invalid: {exc}")
                continue
            if position is not None:
                errors.extend(
                    _check_optional_metric(
                        case,
                        "fh_position",
                        position,
                        case_id=case_id,
                        tol=tol,
                    )
                )
            errors.extend(
                _check_optional_metric(
                    case,
                    "independence_regret",
                    independence_regret(observed, independent),
                    case_id=case_id,
                    tol=tol,
                )
            )
        errors.extend(
            _check_optional_metric(
                case,
                "independence_regret_lower",
                independence_regret(lower, independent),
                case_id=case_id,
                tol=tol,
            )
        )
        errors.extend(
            _check_optional_metric(
                case,
                "independence_regret_upper",
                independence_regret(upper, independent),
                case_id=case_id,
                tol=tol,
            )
        )
    return errors


def _check_optional_metric(
    payload: Mapping[str, Any],
    field: str,
    expected: float,
    *,
    case_id: str,
    tol: float,
) -> list[str]:
    if field not in payload:
        return []
    try:
        actual = float(payload[field])
    except (TypeError, ValueError) as exc:
        return [f"{case_id}.{field} is invalid: {exc}."]
    if abs(actual - expected) > tol:
        return [f"{case_id}.{field} {actual} does not match canonical value {expected}."]
    return []


def _verify_classical_table(path: Path, *, tol: float) -> list[str]:
    errors: list[str] = []
    for row in _read_csv(path):
        scenario = row.get("scenario", "<unknown>")
        try:
            marginals_payload = json.loads(row["marginals_json"])
            marginals = [float(marginals_payload[key]) for key in sorted(marginals_payload)]
            event = row["event"]
            expected_lower, expected_upper = classical_frechet_bounds(marginals, event=event)
            formula_lower = float(row["formula_lower"])
            formula_upper = float(row["formula_upper"])
            atom_lp_lower = float(row["atom_lp_lower"])
            atom_lp_upper = float(row["atom_lp_upper"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Malformed classical table row {scenario}: {exc}")
            continue
        if abs(formula_lower - expected_lower) > tol or abs(formula_upper - expected_upper) > tol:
            errors.append(f"{scenario} formula bounds do not match cc.kernel.frechet_classes.")
        if abs(atom_lp_lower - expected_lower) > tol or abs(atom_lp_upper - expected_upper) > tol:
            errors.append(f"{scenario} atom-LP bounds do not match classical formulas.")
    return errors


def _verify_metric_table(path: Path, *, tol: float) -> list[str]:
    errors: list[str] = []
    for row in _read_csv(path):
        metric = row.get("metric", "<unknown>")
        try:
            inputs = json.loads(row["inputs_json"])
            recorded = float(row["value"])
            expected = _metric_value(str(metric), inputs)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"Malformed metric table row {metric}: {exc}")
            continue
        if expected is None:
            errors.append(f"Metric {metric} unexpectedly returned None.")
            continue
        if abs(recorded - expected) > tol:
            errors.append(f"Metric {metric} value {recorded} does not match canonical value {expected}.")
    return errors


def _metric_value(metric: str, inputs: Mapping[str, Any]) -> float | None:
    if metric == "fh_width":
        return fh_width(float(inputs["lower"]), float(inputs["upper"]))
    if metric == "fh_position":
        return fh_position(
            float(inputs["observed"]),
            float(inputs["lower"]),
            float(inputs["upper"]),
        )
    if metric == "independent_event_probability":
        labels = tuple(_string_list(inputs["labels"], label="metric.labels"))
        query_name = str(inputs["query"])
        if query_name == "union_all":
            query = LinearQuery.union(labels, labels, name=query_name)
        elif query_name == "intersection_all":
            query = LinearQuery.intersection(labels, labels, name=query_name)
        else:
            raise ValueError(f"Unsupported metric query {query_name!r}")
        return independent_event_probability(
            _float_mapping(inputs["marginals"], label="metric.marginals"),
            query,
            labels=labels,
        )
    if metric == "independence_regret":
        return independence_regret(float(inputs["observed"]), float(inputs["independent"]))
    if metric == "cc_gain":
        return cc_gain(
            float(inputs["composition_risk"]),
            _float_mapping(inputs["singleton_failures"], label="metric.singleton_failures"),
        )
    if metric == "cc_shift":
        return cc_shift(
            float(inputs["composition_baseline"]),
            float(inputs["composition_deployed"]),
            _float_mapping(inputs["singleton_baseline"], label="metric.singleton_baseline"),
            _float_mapping(inputs["singleton_deployed"], label="metric.singleton_deployed"),
        )
    raise ValueError(f"Unknown metric {metric!r}")


def _verify_witness_table(path: Path, *, tol: float) -> list[str]:
    errors: list[str] = []
    for row in _read_csv(path):
        case_id = row.get("case_id", "<unknown>")
        endpoint = row.get("endpoint", "<unknown>")
        try:
            passed = row["passed"].lower() == "true"
            probability_sum = float(row["probability_sum"])
            min_probability = float(row["min_probability"])
            query_value = float(row["query_value"])
            reported_bound = float(row["reported_bound"])
            absolute_error = float(row["absolute_error"])
        except (KeyError, ValueError) as exc:
            errors.append(f"Malformed witness verification table row {case_id}.{endpoint}: {exc}")
            continue
        if not passed:
            errors.append(f"Witness verification table marks {case_id}.{endpoint} as failed.")
        if abs(probability_sum - 1.0) > tol:
            errors.append(f"Witness verification table sum for {case_id}.{endpoint} is not 1.")
        if min_probability < -tol:
            errors.append(f"Witness verification table min probability for {case_id}.{endpoint} is negative.")
        if abs(query_value - reported_bound) > tol or absolute_error > tol:
            errors.append(f"Witness verification table query mismatch for {case_id}.{endpoint}.")
    return errors


def _verify_sample_complexity_table(path: Path, *, tol: float) -> list[str]:
    errors: list[str] = []
    for row in _read_csv(path):
        label = f"m={row.get('guardrails')} n={row.get('n_samples')}"
        try:
            num_rates = int(row["num_simultaneous_rates"])
            n_samples = int(row["n_samples"])
            delta = float(row["delta"])
            radius = float(row["hoeffding_radius"])
            epsilon = float(row["target_epsilon"])
            sample_size = int(row["sample_size_for_epsilon"])
        except (KeyError, ValueError) as exc:
            errors.append(f"Malformed sample complexity row {label}: {exc}")
            continue
        expected_radius = simultaneous_bernoulli_radius(n_samples, num_rates, delta)
        expected_sample_size = simultaneous_sample_size(epsilon, num_rates, delta)
        if abs(radius - expected_radius) > tol:
            errors.append(f"Sample complexity radius mismatch for {label}.")
        if sample_size != expected_sample_size:
            errors.append(f"Sample complexity sample-size mismatch for {label}.")
    return errors


def _verify_runtime_scaling_table(path: Path) -> list[str]:
    errors: list[str] = []
    for row in _read_csv(path):
        try:
            guardrails = int(row["guardrails"])
            atom_variables = int(row["atom_lp_variables"])
            singleton_constraints = int(row["singleton_constraints"])
            pairwise_rates = int(row["pairwise_overlap_rates"])
        except (KeyError, ValueError) as exc:
            errors.append(f"Malformed runtime scaling row: {exc}")
            continue
        if row.get("closed_form_marginal_and_or") != "available":
            errors.append(f"Closed-form status should be available for m={guardrails}.")
        if atom_variables != 1 << guardrails:
            errors.append(f"Atom variable count mismatch for m={guardrails}.")
        if singleton_constraints != 2 * guardrails:
            errors.append(f"Singleton constraint count mismatch for m={guardrails}.")
        if pairwise_rates != guardrails * (guardrails - 1) // 2:
            errors.append(f"Pairwise rate count mismatch for m={guardrails}.")
    return errors


def _verify_benchmark_example(payload: Mapping[str, Any]) -> list[str]:
    errors = verify_benchmark_summary(payload)
    if errors:
        return [f"benchmark_example_summary.json: {error}" for error in errors]
    if payload.get("dataset", {}).get("sha256") == "paper-fixture-not-real-model-evidence":
        return []
    return []


def _verify_png_artifacts(artifact_dir: Path) -> list[str]:
    errors: list[str] = []
    for filename in (
        "figure_1_fh_interval.png",
        "figure_2_independence_regret.png",
        "figure_3_correlation_cliff_toy.png",
        "figure_4_runtime_scaling.png",
    ):
        path = artifact_dir / filename
        try:
            with path.open("rb") as handle:
                signature = handle.read(len(PNG_SIGNATURE))
        except OSError as exc:
            errors.append(f"Unable to read {filename}: {exc}")
            continue
        if signature != PNG_SIGNATURE:
            errors.append(f"{filename} is not a valid PNG artifact.")
        if path.stat().st_size <= len(PNG_SIGNATURE):
            errors.append(f"{filename} is unexpectedly empty.")
    return errors


def _cases_by_id(cases: Any, *, label: str) -> dict[str, Mapping[str, Any]]:
    if not isinstance(cases, list):
        raise ArtifactVerificationError(f"{label} cases must be a list.")
    out: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        if not isinstance(case, Mapping):
            raise ArtifactVerificationError(f"{label} contains a non-object case.")
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ArtifactVerificationError(f"{label} contains a case without a string case_id.")
        out[case_id] = case
    return out


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_manifest_payload(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _string_list(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list.")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise TypeError(f"{label} must contain nonempty strings.")
        out.append(item)
    return out


def _float_mapping(value: Any, *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    out: dict[str, float] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{label} keys must be nonempty strings.")
        out[key] = float(item)
    return out


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    return value


def _float_vector(value: Any, *, expected_length: int, label: str) -> np.ndarray[Any, Any]:
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.size != expected_length:
        raise ValueError(f"{label} must be a vector of length {expected_length}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} must contain finite values.")
    return vector


if __name__ == "__main__":
    raise SystemExit(main())
