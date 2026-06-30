#!/usr/bin/env python
# ruff: noqa: I001
"""Generate deterministic paper-facing kernel artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as importlib_metadata
import json
import platform
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from cc.evals.dependence_benchmark import build_summary_from_failure_matrix
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
    bernoulli_rate_count,
    simultaneous_bernoulli_radius,
    simultaneous_sample_size,
)
from cc.kernel.sensitivity import AssumptionSet, LinearQuery, identified_region


ARTIFACT_FILENAMES = (
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
HASHED_FILENAMES = tuple(name for name in ARTIFACT_FILENAMES if name != "manifest.json")
FIXED_REPRODUCIBILITY_TIMESTAMP = "1970-01-01T00:00:00Z"
FAILURE_EVENT_CONVENTION = "Z_i=1 denotes a guardrail failure or unsafe pass."
DEFAULT_OUTPUT_DIR = Path("artifacts/paper")
NUMERIC_TOL = 1.0e-8


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        "--out",
        dest="output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact directory to create or update.",
    )
    args = parser.parse_args()

    command = _command_string(sys.argv)
    generate_artifacts(args.output_dir, generation_command=command)
    print(f"Wrote paper artifacts to {args.output_dir}")
    return 0


def generate_artifacts(output_dir: Path, *, generation_command: str) -> None:
    _prepare_output_dir(output_dir)

    cases = _build_minimal_cases()
    _write_table_1(output_dir / "table_1_classical_frechet_bounds.csv")
    _write_table_2(output_dir / "table_2_metric_examples.csv")
    _write_table_3(output_dir / "table_3_witness_verification.csv", cases)
    _write_table_4(output_dir / "table_4_sample_complexity.csv")
    scaling_rows = _runtime_scaling_rows()
    _write_table_5(output_dir / "table_5_runtime_scaling.csv", scaling_rows)
    _write_figures(output_dir, cases)
    _write_runtime_figure(output_dir / "figure_4_runtime_scaling.png", scaling_rows)

    minimal_bounds = _minimal_bounds_payload(cases)
    minimal_witnesses = _minimal_witnesses_payload(cases)
    environment = _environment_payload(generation_command)
    minimal_bundle = _minimal_bundle_payload(cases)
    benchmark_summary = _benchmark_example_payload()

    _write_json(output_dir / "minimal_bounds.json", minimal_bounds)
    _write_json(output_dir / "minimal_witnesses.json", minimal_witnesses)
    _write_json(output_dir / "minimal_bundle.json", minimal_bundle)
    _write_json(output_dir / "benchmark_example_summary.json", benchmark_summary)
    _write_json(output_dir / "environment.json", environment)
    _write_manifest(output_dir, generation_command=generation_command)


def _build_minimal_cases() -> list[dict[str, Any]]:
    labels = ("input_filter", "semantic_judge")
    marginals = {"input_filter": 0.2, "semantic_judge": 0.35}
    assumptions = _exact_marginal_assumptions(labels, marginals)

    specs = (
        ("minimal_and", "and", 0.12, LinearQuery.intersection(labels, labels, name="AND_failure")),
        ("minimal_or", "or", 0.45, LinearQuery.union(labels, labels, name="OR_failure")),
    )
    cases: list[dict[str, Any]] = []
    for case_id, event, observed, query in specs:
        result = identified_region(query, assumptions)
        independent = independent_event_probability(marginals, query, labels=labels)
        lower_regret = independence_regret(result.lower_bound, independent)
        upper_regret = independence_regret(result.upper_bound, independent)
        cases.append(
            {
                "case_id": case_id,
                "event": event,
                "labels": list(labels),
                "declared_marginals": dict(marginals),
                "assumptions": _assumptions_payload(assumptions),
                "assumptions_hash": result.assumptions_hash,
                "query": _query_payload(query, event=event, events=labels),
                "lower_bound": result.lower_bound,
                "upper_bound": result.upper_bound,
                "fh_width": fh_width(result.lower_bound, result.upper_bound),
                "observed": observed,
                "fh_position": fh_position(observed, result.lower_bound, result.upper_bound),
                "product_baseline": independent,
                "independent_baseline": independent,
                "independence_regret": independence_regret(observed, independent),
                "independence_regret_lower": lower_regret,
                "independence_regret_upper": upper_regret,
                "solver_status": result.solver_status,
                "witnesses": {
                    "lower": {
                        "distribution": _float_list(result.lower_solution),
                        "query_value": float(query.coefficients @ result.lower_solution),
                        "active_constraints": list(result.active_constraints_lower),
                    },
                    "upper": {
                        "distribution": _float_list(result.upper_solution),
                        "query_value": float(query.coefficients @ result.upper_solution),
                        "active_constraints": list(result.active_constraints_upper),
                    },
                },
            }
        )
    return cases


def _exact_marginal_assumptions(
    labels: Sequence[str],
    marginals: Mapping[str, float],
) -> AssumptionSet:
    assumptions = AssumptionSet.empty(
        labels,
        metadata={
            "failure_event_convention": FAILURE_EVENT_CONVENTION,
            "source": "scripts/reproduce_paper.py",
        },
    )
    for label in labels:
        value = float(marginals[label])
        assumptions = assumptions.with_marginal_interval(label, value, value)
    return assumptions


def _write_table_1(path: Path) -> None:
    rows: list[dict[str, Any]] = []
    scenarios = (
        ("two_event_and", ("G0", "G1"), (0.3, 0.7), "and"),
        ("two_event_or", ("G0", "G1"), (0.3, 0.7), "or"),
        ("three_event_and", ("G0", "G1", "G2"), (0.2, 0.5, 0.8), "and"),
        ("three_event_or", ("G0", "G1", "G2"), (0.2, 0.5, 0.8), "or"),
    )
    for scenario, labels, marginal_values, event in scenarios:
        marginals = dict(zip(labels, marginal_values, strict=True))
        assumptions = _exact_marginal_assumptions(labels, marginals)
        query = (
            LinearQuery.intersection(labels, labels, name=f"{scenario}_query")
            if event == "and"
            else LinearQuery.union(labels, labels, name=f"{scenario}_query")
        )
        result = identified_region(query, assumptions)
        formula_lower, formula_upper = classical_frechet_bounds(marginal_values, event=event)
        rows.append(
            {
                "scenario": scenario,
                "event": event,
                "marginals_json": _compact_json(marginals),
                "formula_lower": formula_lower,
                "formula_upper": formula_upper,
                "atom_lp_lower": result.lower_bound,
                "atom_lp_upper": result.upper_bound,
                "fh_width": fh_width(result.lower_bound, result.upper_bound),
            }
        )
    _write_csv(
        path,
        (
            "scenario",
            "event",
            "marginals_json",
            "formula_lower",
            "formula_upper",
            "atom_lp_lower",
            "atom_lp_upper",
            "fh_width",
        ),
        rows,
    )


def _write_table_2(path: Path) -> None:
    labels = ("A", "B")
    union_query = LinearQuery.union(labels, labels, name="A_or_B")
    metric_rows = [
        (
            "fh_width",
            {"lower": 0.1, "upper": 0.4},
            fh_width(0.1, 0.4),
        ),
        (
            "fh_position",
            {"observed": 0.25, "lower": 0.1, "upper": 0.4},
            fh_position(0.25, 0.1, 0.4),
        ),
        (
            "independent_event_probability",
            {
                "labels": list(labels),
                "marginals": {"A": 0.2, "B": 0.3},
                "query": "union_all",
            },
            independent_event_probability({"A": 0.2, "B": 0.3}, union_query, labels=labels),
        ),
        (
            "independence_regret",
            {"observed": 0.5, "independent": 0.44},
            independence_regret(0.5, 0.44),
        ),
        (
            "cc_gain",
            {"composition_risk": 0.18, "singleton_failures": {"A": 0.04, "B": 0.06}},
            cc_gain(0.18, {"A": 0.04, "B": 0.06}),
        ),
        (
            "cc_shift",
            {
                "composition_baseline": 0.1,
                "composition_deployed": 0.16,
                "singleton_baseline": {"A": 0.08, "B": 0.11},
                "singleton_deployed": {"A": 0.10, "B": 0.08},
            },
            cc_shift(
                0.1,
                0.16,
                {"A": 0.08, "B": 0.11},
                {"A": 0.10, "B": 0.08},
            ),
        ),
    ]
    rows = [
        {"metric": metric, "inputs_json": _compact_json(inputs), "value": value}
        for metric, inputs, value in metric_rows
    ]
    _write_csv(path, ("metric", "inputs_json", "value"), rows)


def _write_table_3(path: Path, cases: Sequence[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for case in cases:
        query = np.asarray(case["query"]["coefficients"], dtype=float)
        for endpoint in ("lower", "upper"):
            witness = case["witnesses"][endpoint]
            distribution = np.asarray(witness["distribution"], dtype=float)
            query_value = float(query @ distribution)
            reported = float(case[f"{endpoint}_bound"])
            rows.append(
                {
                    "case_id": case["case_id"],
                    "endpoint": endpoint,
                    "probability_sum": float(np.sum(distribution)),
                    "min_probability": float(np.min(distribution)),
                    "query_value": query_value,
                    "reported_bound": reported,
                    "absolute_error": abs(query_value - reported),
                    "passed": abs(query_value - reported) <= NUMERIC_TOL
                    and float(np.min(distribution)) >= -NUMERIC_TOL
                    and abs(float(np.sum(distribution)) - 1.0) <= NUMERIC_TOL,
                }
            )
    _write_csv(
        path,
        (
            "case_id",
            "endpoint",
            "probability_sum",
            "min_probability",
            "query_value",
            "reported_bound",
            "absolute_error",
            "passed",
        ),
        rows,
    )


def _write_table_4(path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for guardrails in (2, 4, 8):
        num_rates = bernoulli_rate_count(guardrails, include_pairwise=True)
        for n_samples in (100, 500, 2000):
            delta = 0.05
            epsilon = 0.05
            rows.append(
                {
                    "guardrails": guardrails,
                    "num_simultaneous_rates": num_rates,
                    "n_samples": n_samples,
                    "delta": delta,
                    "hoeffding_radius": simultaneous_bernoulli_radius(
                        n_samples,
                        num_rates,
                        delta,
                    ),
                    "target_epsilon": epsilon,
                    "sample_size_for_epsilon": simultaneous_sample_size(
                        epsilon,
                        num_rates,
                        delta,
                    ),
                }
            )
    _write_csv(
        path,
        (
            "guardrails",
            "num_simultaneous_rates",
            "n_samples",
            "delta",
            "hoeffding_radius",
            "target_epsilon",
            "sample_size_for_epsilon",
        ),
        rows,
    )


def _runtime_scaling_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for guardrails in (2, 4, 6, 8, 10, 12):
        atom_variables = 1 << guardrails
        rows.append(
            {
                "guardrails": guardrails,
                "closed_form_marginal_and_or": "available",
                "atom_lp_variables": atom_variables,
                "simplex_equality_constraints": 1,
                "singleton_constraints": 2 * guardrails,
                "pairwise_overlap_rates": guardrails * (guardrails - 1) // 2,
            }
        )
    return rows


def _write_table_5(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _write_csv(
        path,
        (
            "guardrails",
            "closed_form_marginal_and_or",
            "atom_lp_variables",
            "simplex_equality_constraints",
            "singleton_constraints",
            "pairwise_overlap_rates",
        ),
        rows,
    )


def _write_figures(output_dir: Path, cases: Sequence[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 2.8), constrained_layout=True)
    y_positions = np.arange(len(cases))
    for y_position, case in zip(y_positions, cases, strict=True):
        lower = float(case["lower_bound"])
        upper = float(case["upper_bound"])
        observed = float(case["observed"])
        ax.hlines(y_position, lower, upper, color="#3B6EA8", linewidth=4)
        ax.plot([lower, upper], [y_position, y_position], "o", color="#153B50", markersize=5)
        ax.plot(observed, y_position, "D", color="#C44536", markersize=5)
    ax.set_yticks(y_positions, [str(case["event"]).upper() for case in cases])
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Failure probability")
    ax.set_title("Sharp Frechet-Hoeffding intervals")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(
        output_dir / "figure_1_fh_interval.png",
        dpi=150,
        metadata={"Software": "cc-framework reproduce_paper.py"},
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 2.8), constrained_layout=True)
    labels = [str(case["event"]).upper() for case in cases]
    regrets = [float(case["independence_regret"]) for case in cases]
    colors = ["#C44536" if value >= 0.0 else "#3B6EA8" for value in regrets]
    ax.bar(labels, regrets, color=colors, width=0.55)
    ax.axhline(0.0, color="#222222", linewidth=1)
    ax.set_ylabel("Observed - independent")
    ax.set_title("Independence regret examples")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(
        output_dir / "figure_2_independence_regret.png",
        dpi=150,
        metadata={"Software": "cc-framework reproduce_paper.py"},
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    p_left = 0.2
    p_right = 0.35
    joint_lower, joint_upper = classical_frechet_bounds((p_left, p_right), event="and")
    product_joint = p_left * p_right
    joint_values = np.linspace(joint_lower, joint_upper, 101)
    dependence_shift = joint_values - product_joint
    ax.plot(dependence_shift, joint_values, color="#3B6EA8", linewidth=2.2, label="AND")
    ax.plot(
        dependence_shift,
        p_left + p_right - joint_values,
        color="#C44536",
        linewidth=2.2,
        label="OR",
    )
    ax.axvline(0.0, color="#222222", linewidth=1.0, alpha=0.8)
    ax.scatter([0.0], [product_joint], color="#3B6EA8", s=24, zorder=3)
    ax.scatter([0.0], [p_left + p_right - product_joint], color="#C44536", s=24, zorder=3)
    ax.set_xlabel("P(A and B) - P(A)P(B)")
    ax.set_ylabel("Composition failure probability")
    ax.set_title("Toy dependence sweep")
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.25)
    fig.savefig(
        output_dir / "figure_3_correlation_cliff_toy.png",
        dpi=150,
        metadata={"Software": "cc-framework reproduce_paper.py"},
    )
    plt.close(fig)


def _write_runtime_figure(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    guardrails = [int(row["guardrails"]) for row in rows]
    atom_variables = [int(row["atom_lp_variables"]) for row in rows]
    fig, ax = plt.subplots(figsize=(6.0, 3.0), constrained_layout=True)
    ax.plot(guardrails, atom_variables, marker="o", color="#3B6EA8", linewidth=2.2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("Number of guardrails")
    ax.set_ylabel("Explicit atom variables")
    ax.set_title("General atom-LP scaling")
    ax.grid(alpha=0.25, which="both")
    fig.savefig(
        path,
        dpi=150,
        metadata={"Software": "cc-framework reproduce_paper.py"},
    )
    plt.close(fig)


def _benchmark_example_payload() -> dict[str, Any]:
    labels = ("llama_guard", "keyword_blocker")
    failure_matrix = np.asarray(
        [
            [1, 0],
            [0, 1],
            [0, 0],
            [1, 1],
        ],
        dtype=int,
    )
    records = [
        {
            "row_index": index,
            "row_id": row_id,
            "label": "harmful",
            "failure_indicators": {
                labels[column]: int(failure_matrix[index, column])
                for column in range(len(labels))
            },
        }
        for index, row_id in enumerate(("ex1", "ex2", "ex3", "ex4"))
    ]
    return build_summary_from_failure_matrix(
        labels,
        failure_matrix,
        records=records,
        adapter_versions={
            "llama_guard": "mock-llama-guard@paper-fixture",
            "keyword_blocker": "deterministic-keyword-v1",
        },
        dataset_id="tests/fixtures/dependence_benchmark_harmful.csv",
        dataset_sha256="paper-fixture-not-real-model-evidence",
        run_id="paper-fixture",
    )


def _minimal_bounds_payload(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "cc.paper.bounds.v1",
        "failure_event_convention": FAILURE_EVENT_CONVENTION,
        "cases": [
            {
                "case_id": case["case_id"],
                "event": case["event"],
                "labels": case["labels"],
                "declared_marginals": case["declared_marginals"],
                "query": case["query"],
                "lower_bound": case["lower_bound"],
                "upper_bound": case["upper_bound"],
                "fh_width": case["fh_width"],
                "observed": case["observed"],
                "fh_position": case["fh_position"],
                "product_baseline": case["product_baseline"],
                "independent_baseline": case["independent_baseline"],
                "independence_regret": case["independence_regret"],
                "independence_regret_lower": case["independence_regret_lower"],
                "independence_regret_upper": case["independence_regret_upper"],
                "assumptions_hash": case["assumptions_hash"],
            }
            for case in cases
        ],
    }


def _minimal_witnesses_payload(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "cc.paper.witnesses.v1",
        "atom_order": "little_endian",
        "tolerance": NUMERIC_TOL,
        "cases": [
            {
                "case_id": case["case_id"],
                "event": case["event"],
                "labels": case["labels"],
                "declared_marginals": case["declared_marginals"],
                "assumptions": case["assumptions"],
                "assumptions_hash": case["assumptions_hash"],
                "query": case["query"],
                "bounds": {
                    "lower": case["lower_bound"],
                    "upper": case["upper_bound"],
                },
                "witnesses": case["witnesses"],
            }
            for case in cases
        ],
    }


def _minimal_bundle_payload(cases: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "cc.paper.bundle.v1",
        "failure_event_convention": FAILURE_EVENT_CONVENTION,
        "artifact_files": list(ARTIFACT_FILENAMES),
        "bounds_file": "minimal_bounds.json",
        "witnesses_file": "minimal_witnesses.json",
        "environment_file": "environment.json",
        "witness_verification_table": "table_3_witness_verification.csv",
        "cases": [
            {
                "case_id": case["case_id"],
                "event": case["event"],
                "lower_bound": case["lower_bound"],
                "upper_bound": case["upper_bound"],
                "fh_width": case["fh_width"],
                "product_baseline": case["product_baseline"],
                "independence_regret": case["independence_regret"],
                "independence_regret_lower": case["independence_regret_lower"],
                "independence_regret_upper": case["independence_regret_upper"],
            }
            for case in cases
        ],
    }


def _environment_payload(generation_command: str) -> dict[str, Any]:
    return {
        "schema_version": "cc.paper.environment.v1",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "package_version": _package_version("cc-framework"),
        "dependency_versions": {
            package: _package_version(package)
            for package in ("numpy", "scipy", "matplotlib", "jsonschema")
        },
        "timestamp_policy": "fixed reproducibility timestamp; wall-clock time omitted",
        "generation_timestamp": FIXED_REPRODUCIBILITY_TIMESTAMP,
        "generation_command": generation_command,
        "random_seed_policy": "no random sampling is used; deterministic examples only",
    }


def _write_manifest(output_dir: Path, *, generation_command: str) -> None:
    package_version = _package_version("cc-framework")
    manifest: dict[str, Any] = {
        "schema_version": "cc.paper.manifest.v1",
        "generation_command": generation_command,
        "package_version": package_version,
        "required_files": list(ARTIFACT_FILENAMES),
        "files": [
            {
                "filename": filename,
                "sha256": _sha256_file(output_dir / filename),
                "bytes": (output_dir / filename).stat().st_size,
            }
            for filename in HASHED_FILENAMES
        ],
    }
    manifest["manifest_payload_sha256"] = _hash_manifest_payload(manifest)
    _write_json(output_dir / "manifest.json", manifest)


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ARTIFACT_FILENAMES:
        path = output_dir / filename
        if path.exists():
            path.unlink()


def _assumptions_payload(assumptions: AssumptionSet) -> dict[str, Any]:
    return {
        "guardrails": list(assumptions.guardrails),
        "metadata": dict(assumptions.metadata),
        "constraints": [
            {
                "name": constraint.name,
                "sense": constraint.sense,
                "rhs": constraint.rhs,
                "coefficients": _float_list(constraint.coefficients),
            }
            for constraint in assumptions.constraints
        ],
    }


def _query_payload(query: LinearQuery, *, event: str, events: Sequence[str]) -> dict[str, Any]:
    return {
        "name": query.name,
        "event": event,
        "events": list(events),
        "coefficients": _float_list(query.coefficients),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: _csv_value(row.get(key))
                    for key in fieldnames
                }
            )


def _csv_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".17g")
    if value is None:
        return ""
    return str(value)


def _float_list(values: Sequence[float] | np.ndarray[Any, Any]) -> list[float]:
    return [float(value) for value in values]


def _compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


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


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None


def _command_string(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in argv)


if __name__ == "__main__":
    raise SystemExit(main())
