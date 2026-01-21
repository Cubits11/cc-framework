from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from experiments.fh_atlas.config import FHAtlasConfig
from experiments.fh_atlas.manifest import ManifestEntry, build_manifest, hash_file, write_manifest
from experiments.fh_atlas.plots import (
    plot_cc_regime_heatmap,
    plot_cii_distribution,
    plot_fh_envelope,
    plot_identifiability_map,
    plot_more_rails_scaling,
)
from experiments.fh_atlas.simulate_dependence import simulate_dependence
from theory.fh_bounds import (
    compute_composability_interference_index,
    default_cc_regime_thresholds,
    independence_parallel_and_j,
    independence_serial_or_j,
    parallel_and_composition_bounds,
    propagate_marginal_uncertainty_to_composed_bounds,
    serial_or_composition_bounds,
    wilson_score_interval,
)


def _hash_config(config: FHAtlasConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _valid_theta_values(family: str, theta_values: list[float]) -> list[float]:
    if family == "clayton":
        return [theta for theta in theta_values if theta > 0]
    if family == "gumbel":
        return [theta for theta in theta_values if theta >= 1.0]
    if family == "frank":
        return [theta for theta in theta_values if abs(theta) > 1e-12]
    return theta_values


def run_fh_atlas(config: FHAtlasConfig) -> Path:
    config.validate()
    rng = np.random.default_rng(config.seed)

    run_id = datetime.utcnow().strftime("fh_atlas_%Y%m%d_%H%M%S")
    output_root = Path(config.output_dir) / run_id
    metrics_dir = output_root / "metrics"
    plots_dir = output_root / "plots"
    certificate_dir = output_root / "certificate"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    certificate_dir.mkdir(parents=True, exist_ok=True)

    table_rows: list[dict[str, object]] = []
    manifest_entries: list[ManifestEntry] = []

    for composition_type in config.composition_types:
        for k in config.k_values:
            for miss_rate in config.miss_rate_grid:
                for fpr_rate in config.fpr_rate_grid:
                    miss_rates = [miss_rate] * k
                    fpr_rates = [fpr_rate] * k
                    if composition_type == "serial_or":
                        bounds = serial_or_composition_bounds(miss_rates, fpr_rates)
                        j_independence = independence_serial_or_j(
                            [1.0 - m for m in miss_rates],
                            fpr_rates,
                        )
                    else:
                        bounds = parallel_and_composition_bounds(miss_rates, fpr_rates)
                        j_independence = independence_parallel_and_j(
                            [1.0 - m for m in miss_rates],
                            fpr_rates,
                        )

                    for family in config.copula_families:
                        thetas = _valid_theta_values(family, list(config.theta_grid))
                        if not thetas:
                            continue
                        observed_js: list[float] = []
                        observed_ci: list[tuple[float, float]] = []
                        cii_values: list[float] = []
                        certificate_payload: dict[str, object] | None = None

                        for theta in thetas:
                            for sample_size in config.sample_sizes:
                                sim = simulate_dependence(
                                    tprs=[1.0 - m for m in miss_rates],
                                    fprs=fpr_rates,
                                    composition_type=composition_type,
                                    family=family,
                                    theta=theta,
                                    sample_size=sample_size,
                                    rng=rng,
                                )
                                observed_js.append(sim.observed_j)

                                tpr_ci = wilson_score_interval(
                                    successes=int(sim.observed_tpr * sample_size),
                                    trials=sample_size,
                                )
                                fpr_ci = wilson_score_interval(
                                    successes=int(sim.observed_fpr * sample_size),
                                    trials=sample_size,
                                )
                                observed_ci.append((tpr_ci[0] - fpr_ci[1], tpr_ci[1] - fpr_ci[0]))

                                cii = compute_composability_interference_index(
                                    observed_j=sim.observed_j,
                                    bounds=bounds,
                                    individual_tprs=[1.0 - m for m in miss_rates],
                                    individual_fprs=fpr_rates,
                                )
                                cii_values.append(float(cii["cii"]))

                                if certificate_payload is None:
                                    certificate_payload = (
                                        propagate_marginal_uncertainty_to_composed_bounds(
                                            tp_counts=sim.tp_counts,
                                            fn_counts=sim.fn_counts,
                                            fp_counts=sim.fp_counts,
                                            tn_counts=sim.tn_counts,
                                            composition_type=composition_type,
                                            alpha=0.05,
                                        )
                                    )

                                table_rows.append(
                                    {
                                        "scenario": f"{composition_type}_{family}",
                                        "k": k,
                                        "miss_rate": miss_rate,
                                        "fpr_rate": fpr_rate,
                                        "theta": theta,
                                        "sample_size": sample_size,
                                        "j_lower": bounds.j_lower,
                                        "j_upper": bounds.j_upper,
                                        "j_independence": j_independence,
                                        "observed_j": sim.observed_j,
                                        "cii": float(cii["cii"]),
                                    }
                                )

                        scenario_id = (
                            f"{composition_type}_{family}_k{k}_"
                            f"miss{miss_rate:.2f}_fpr{fpr_rate:.2f}"
                        )
                        metrics_payload = {
                            "scenario_id": scenario_id,
                            "composition_type": composition_type,
                            "family": family,
                            "k": k,
                            "miss_rate": miss_rate,
                            "fpr_rate": fpr_rate,
                            "theta_grid": thetas,
                            "j_bounds": {"lower": bounds.j_lower, "upper": bounds.j_upper},
                            "j_independence": j_independence,
                            "observed_j": observed_js,
                            "observed_j_ci": observed_ci,
                            "cii": cii_values,
                            "cc_regime": bounds.classify_regime(
                                threshold_policy=default_cc_regime_thresholds()
                            ),
                        }

                        metrics_path = metrics_dir / f"scenario_{scenario_id}.json"
                        metrics_path.write_text(
                            json.dumps(metrics_payload, indent=2), encoding="utf-8"
                        )
                        manifest_entries.append(
                            ManifestEntry(
                                path=str(metrics_path.relative_to(output_root)),
                                sha256=hash_file(metrics_path),
                                size_bytes=metrics_path.stat().st_size,
                                description="Scenario metrics",
                            )
                        )

                        title = (
                            f"FH envelope ({composition_type}, {family})\n"
                            f"k={k}, miss={miss_rate:.2f}, fpr={fpr_rate:.2f}"
                        )
                        fh_plot_path = plot_fh_envelope(
                            plots_dir,
                            scenario_id,
                            thetas,
                            observed_js,
                            observed_ci,
                            bounds,
                            j_independence,
                            title,
                        )
                        cii_plot_path = plot_cii_distribution(
                            plots_dir, scenario_id, thetas, cii_values
                        )

                        cc_plot_path = plot_cc_regime_heatmap(
                            plots_dir,
                            f"{scenario_id}_heatmap",
                            config.miss_rate_grid,
                            config.fpr_rate_grid,
                            composition_type,
                        )
                        ident_plot_path = plot_identifiability_map(
                            plots_dir,
                            f"{scenario_id}_identifiability",
                            config.miss_rate_grid,
                            config.fpr_rate_grid,
                            composition_type,
                        )
                        scaling_plot_path = plot_more_rails_scaling(
                            plots_dir,
                            f"{scenario_id}_scaling",
                            config.k_values,
                            miss_rate,
                            fpr_rate,
                            composition_type,
                        )

                        for plot_path in [
                            fh_plot_path,
                            cii_plot_path,
                            cc_plot_path,
                            ident_plot_path,
                            scaling_plot_path,
                        ]:
                            for artifact in [
                                plot_path,
                                plot_path.with_suffix(".pdf"),
                                plot_path.with_suffix(".json"),
                            ]:
                                manifest_entries.append(
                                    ManifestEntry(
                                        path=str(artifact.relative_to(output_root)),
                                        sha256=hash_file(artifact),
                                        size_bytes=artifact.stat().st_size,
                                        description=f"Plot artifact for {scenario_id}",
                                    )
                                )

                        if certificate_payload is not None:
                            certificate_path = certificate_dir / "certificate.json"
                            certificate_path.write_text(
                                json.dumps(certificate_payload, indent=2),
                                encoding="utf-8",
                            )
                            if not any(
                                entry.path == str(certificate_path.relative_to(output_root))
                                for entry in manifest_entries
                            ):
                                manifest_entries.append(
                                    ManifestEntry(
                                        path=str(certificate_path.relative_to(output_root)),
                                        sha256=hash_file(certificate_path),
                                        size_bytes=certificate_path.stat().st_size,
                                        description="Certified J lower bound",
                                    )
                                )

    table_path = output_root / "tables.csv"
    with table_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)
    manifest_entries.append(
        ManifestEntry(
            path=str(table_path.relative_to(output_root)),
            sha256=hash_file(table_path),
            size_bytes=table_path.stat().st_size,
            description="Scenario table",
        )
    )

    config_hash = _hash_config(config)
    manifest = build_manifest(
        run_id=run_id,
        output_dir=output_root,
        config_hash=config_hash,
        entries=manifest_entries,
    )
    manifest_path = output_root / "manifest.json"
    write_manifest(manifest_path, manifest)

    return output_root
