from __future__ import annotations
"""
Correlation Cliff — Run-All Orchestrator
=======================================
This script orchestrates the full correlation cliff experiment pipeline:
  - Loads and validates YAML config
  - Resolves marginals + matrix of runs (rule × path combos)
  - Computes population curves (coarse, then refined if enabled)
  - Runs finite-sample simulation (optional)
  - Estimates thresholds (population + empirical)
  - Renders publication figures (optional)
  - Writes reproducibility manifest (config, env, hashes)

Supports both legacy single-run schema and enterprise matrix schema.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
# ----------------------------
# Local imports
# ----------------------------
def _import_local():
    """
    Import local modules without hiding real bugs.

    Only ImportError triggers script-style fallback. Any other exception indicates
    a genuine defect (syntax error, runtime error at import time, etc.) and must
    surface immediately.
    """
    try:
        from .simulate import SimConfig, simulate_grid, summarize_simulation, p11_from_path # type: ignore
        from .figures import FigureStyle, make_all_figures # type: ignore
        from .theory import ( # type: ignore
            TwoWorldMarginals,
            WorldMarginals,
            compute_fh_jc_envelope,
            joint_cells_from_marginals,
            pC_from_joint,
            phi_from_joint,
            kendall_tau_a_from_joint,
        )
        return (
            SimConfig,
            simulate_grid,
            summarize_simulation,
            p11_from_path,
            FigureStyle,
            make_all_figures,
            TwoWorldMarginals,
            WorldMarginals,
            compute_fh_jc_envelope,
            joint_cells_from_marginals,
            pC_from_joint,
            phi_from_joint,
            kendall_tau_a_from_joint,
        )
    except ImportError:
        from simulate import SimConfig, simulate_grid, summarize_simulation, p11_from_path # type: ignore
        from figures import FigureStyle, make_all_figures # type: ignore
        from theory import ( # type: ignore
            TwoWorldMarginals,
            WorldMarginals,
            compute_fh_jc_envelope,
            joint_cells_from_marginals,
            pC_from_joint,
            phi_from_joint,
            kendall_tau_a_from_joint,
        )
        return (
            SimConfig,
            simulate_grid,
            summarize_simulation,
            p11_from_path,
            FigureStyle,
            make_all_figures,
            TwoWorldMarginals,
            WorldMarginals,
            compute_fh_jc_envelope,
            joint_cells_from_marginals,
            pC_from_joint,
            phi_from_joint,
            kendall_tau_a_from_joint,
        )


(
    SimConfig,
    simulate_grid,
    summarize_simulation,
    p11_from_path,
    FigureStyle,
    make_all_figures,
    TwoWorldMarginals,
    WorldMarginals,
    compute_fh_jc_envelope,
    joint_cells_from_marginals,
    pC_from_joint,
    phi_from_joint,
    kendall_tau_a_from_joint,
) = _import_local()
# ----------------------------
# Helpers
# ----------------------------
LOG = logging.getLogger("correlation_cliff.run_all")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)
def _now_stamp() -> str:
    # Deterministic enough for file naming; not used for RNG.
    return datetime.now().strftime("%Y%m%d_%H%M%S")
def _atomic_write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    tmp.replace(path)


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")
def _interp_root(x: np.ndarray, y: np.ndarray, target: float) -> Optional[float]:
    if len(x) < 2:
        return None
    for i in range(len(x) - 1):
        y0, y1 = float(y[i]), float(y[i + 1])
        if y0 == target:
            return float(x[i])
        if (y0 - target) * (y1 - target) <= 0:
            if y1 == y0:
                return float(x[i])
            t = (target - y0) / (y1 - y0)
            return float(x[i] + t * (x[i + 1] - x[i]))
    return None
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _maybe_git_commit(repo_root: Path) -> Optional[str]:
    """
    Best-effort git commit capture. Returns None if not available.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:
        return None


def _get(d: Mapping[str, Any], path: str, default: Any = None) -> Any:
    """
    Safe nested get using dot-separated path.
    """
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _require_mapping(x: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(x, Mapping):
        raise ValueError(f"Expected mapping for '{name}', got {type(x).__name__}")
    return x


def _validate_probability(p: float, name: str) -> float:
    try:
        v = float(p)
    except Exception as e:
        raise ValueError(f"{name} must be a float, got {p!r}") from e
    if not (0.0 <= v <= 1.0):
        raise ValueError(f"{name} must be in [0,1], got {v}")
    return v


def _linspace_grid(start: float, stop: float, num: int) -> List[float]:
    if num < 2:
        raise ValueError(f"lambda grid num must be >=2, got {num}")
    a = float(start)
    b = float(stop)
    if not (0.0 <= a <= 1.0 and 0.0 <= b <= 1.0):
        raise ValueError(f"lambda grid start/stop must be in [0,1], got {a},{b}")
    return [float(x) for x in np.linspace(a, b, num=int(num), dtype=float).tolist()]


@dataclass(frozen=True)
class RunSpec:
    """
    Fully resolved “one run” spec (one rule × one dependence path).
    """

    rule: str
    path: str
    path_params: Dict[str, Any]
    lambdas_coarse: List[float]
    refine_enabled: bool
    refine_half_width: float
    refine_num: int
    refine_method: str # "uniform" | "none" (reserved for adaptive later)


def _resolve_runs(cfg: Mapping[str, Any]) -> Tuple[TwoWorldMarginals, List[RunSpec], Dict[str, Any]]:
    """
    Resolve a pipeline config into:
    - TwoWorldMarginals
    - list of RunSpec (rule × path combinations)
    - normalized "meta" dict (useful for manifests)

    Supports two schemas:
    A) "SimConfig-like" direct keys (legacy):
    marginals, rule, lambdas/lambda_grid, n, n_reps, seed, path, path_params

    B) "Pipeline" schema (enterprise):
    marginals
    composition.primary_rule + composition.sensitivity_rules
    dependence_paths.primary + dependence_paths.sensitivity
    """
    md = _require_mapping(cfg.get("marginals", {}), "marginals")
    w0 = _require_mapping(md.get("w0", {}), "marginals.w0")
    w1 = _require_mapping(md.get("w1", {}), "marginals.w1")
    marg = TwoWorldMarginals(
        w0=WorldMarginals(pA=_validate_probability(w0["pA"], "marginals.w0.pA"), pB=_validate_probability(w0["pB"], "marginals.w0.pB")),
        w1=WorldMarginals(pA=_validate_probability(w1["pA"], "marginals.w1.pA"), pB=_validate_probability(w1["pB"], "marginals.w1.pB")),
    )

    # Determine if pipeline schema exists
    has_pipeline = isinstance(cfg.get("dependence_paths"), Mapping) or isinstance(cfg.get("composition"), Mapping)

    rules: List[str] = []
    paths: List[Tuple[str, Dict[str, Any], Mapping[str, Any]]] = [] # (path, params, primary_cfg_for_refine)

    if has_pipeline:
        comp = _require_mapping(cfg.get("composition", {}), "composition")
        primary_rule = str(comp.get("primary_rule", "OR")).upper()
        if primary_rule not in ("OR", "AND"):
            raise ValueError(f"composition.primary_rule must be OR/AND, got {primary_rule}")
        rules.append(primary_rule)
        for r in comp.get("sensitivity_rules", []) or []:
            rr = str(r).upper()
            if rr not in ("OR", "AND"):
                raise ValueError(f"Invalid sensitivity rule: {rr}")
            if rr not in rules:
                rules.append(rr)

        dep = _require_mapping(cfg.get("dependence_paths", {}), "dependence_paths")
        primary = _require_mapping(dep.get("primary", {}), "dependence_paths.primary")
        primary_type = str(primary.get("type", "fh_linear"))
        primary_params = dict(primary.get("path_params", {})) if isinstance(primary.get("path_params", {}), Mapping) else {}
        # also accept direct params like gamma/k/ppf_clip_eps for convenience
        for k in ("gamma", "k", "ppf_clip_eps"):
            if k in primary and k not in primary_params:
                primary_params[k] = primary[k]
        paths.append((primary_type, primary_params, primary))

        for item in dep.get("sensitivity", []) or []:
            if not isinstance(item, Mapping):
                raise ValueError("dependence_paths.sensitivity items must be mappings")
            t = str(item.get("type", "fh_linear"))
            pp: Dict[str, Any] = {}
            for k in ("gamma", "k", "ppf_clip_eps"):
                if k in item:
                    pp[k] = item[k]
            # allow nested path_params too
            if isinstance(item.get("path_params"), Mapping):
                pp.update(dict(item["path_params"]))
            paths.append((t, pp, primary))

        # Coarse grid from pipeline schema
        lg = primary.get("lambda_grid_coarse") or primary.get("lambda_grid") or {}
        lgm = _require_mapping(lg, "dependence_paths.primary.lambda_grid_coarse")
        start = float(lgm.get("start", 0.0))
        stop = float(lgm.get("stop", 1.0))
        num = int(lgm.get("num", 21))
        lambdas_coarse = _linspace_grid(start, stop, num)

        # refine config (optional)
        ref = primary.get("refine") or {}
        refm = _require_mapping(ref, "dependence_paths.primary.refine")
        refine_enabled = bool(refm.get("enabled", False))
        refine_half_width = float(refm.get("half_width", 0.08))
        refine_num = int(refm.get("num", 401))
        refine_method = str(refm.get("method", "uniform"))
    else:
        # legacy schema: exactly one run
        rule = str(cfg.get("rule", "OR")).upper()
        if rule not in ("OR", "AND"):
            raise ValueError(f"rule must be OR/AND, got {rule}")
        rules = [rule]

        path = str(cfg.get("path", "fh_linear"))
        path_params = dict(cfg.get("path_params", {})) if isinstance(cfg.get("path_params", {}), Mapping) else {}
        paths = [(path, path_params, {})]

        if "lambdas" in cfg:
            lambdas_coarse = [float(x) for x in (cfg.get("lambdas") or [])]
        else:
            lg = _require_mapping(cfg.get("lambda_grid", {"num": 21}), "lambda_grid")
            lambdas_coarse = _linspace_grid(float(lg.get("start", 0.0)), float(lg.get("stop", 1.0)), int(lg.get("num", 21)))

        refine_enabled = False
        refine_half_width = 0.08
        refine_num = 401
        refine_method = "uniform"

    # Produce cartesian product of rules × paths
    runs: List[RunSpec] = []
    for r in rules:
        for (ptype, pparams, primary_cfg_for_refine) in paths:
            runs.append(
                RunSpec(
                    rule=r,
                    path=ptype,
                    path_params=dict(pparams),
                    lambdas_coarse=list(lambdas_coarse),
                    refine_enabled=bool(refine_enabled),
                    refine_half_width=float(refine_half_width),
                    refine_num=int(refine_num),
                    refine_method=str(refine_method),
                )
            )

    meta = {
        "schema_detected": "pipeline" if has_pipeline else "legacy",
        "rules": rules,
        "paths": [p[0] for p in paths],
        "coarse_grid_n": int(len(lambdas_coarse)),
        "refine_enabled": bool(refine_enabled),
        "refine_half_width": float(refine_half_width),
        "refine_num": int(refine_num),
        "refine_method": str(refine_method),
    }
    return marg, runs, meta


def _resolve_output_base(cfg: Mapping[str, Any], config_path: Path) -> Tuple[Path, Dict[str, Any]]:
    """
    Resolve output base dir and per-run subdir policy.
    Supports:
    - output.out_dir (preferred)
    - output_dir (legacy)
    - default: experiments/correlation_cliff/artifacts/<stamp>
    """
    out_dir_cfg = _get(cfg, "output.out_dir", None)
    out_dir_legacy = cfg.get("output_dir", None)
    overwrite = bool(_get(cfg, "output.overwrite", False)) if isinstance(_get(cfg, "output", None), Mapping) else bool(cfg.get("overwrite", False))
    save_hashes = bool(_get(cfg, "output.save_hashes", True)) if isinstance(_get(cfg, "output", None), Mapping) else bool(cfg.get("save_hashes", False))

    if out_dir_cfg:
        base = Path(str(out_dir_cfg))
    elif out_dir_legacy:
        base = Path(str(out_dir_legacy))
    else:
        base = config_path.resolve().parent / "artifacts" / _now_stamp()

    policy = {"overwrite": overwrite, "save_hashes": save_hashes}
    return base, policy


def _safe_subdir_name(x: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in x)


# ----------------------------
# Population curve computation (path-aware)
# ----------------------------
def population_curve_from_path(cfg: SimConfig) -> pd.DataFrame:
    """
    Compute the population (infinite-sample) curve for the configured dependence path.
    This is path-aware and includes FH envelopes and dependence summaries (phi, tau).
    """
    rows: list[Dict[str, Any]] = []
    for lam in cfg.lambdas:
        row: Dict[str, Any] = {"lambda": float(lam), "rule": cfg.rule, "path": cfg.path}
        # Population feasibility envelope for JC (path-independent; FH feasibility only)
        jmin, jmax = compute_fh_jc_envelope(cfg.marginals, cfg.rule)
        row["JC_env_min"] = float(jmin)
        row["JC_env_max"] = float(jmax)
        # Per-world: construct joint + compute pop overlays
        for w, wm in ((0, cfg.marginals.w0), (1, cfg.marginals.w1)):
            row[f"pA_true_w{w}"] = float(wm.pA)
            row[f"pB_true_w{w}"] = float(wm.pB)
            # Choose p11 per configured path
            p11, meta = p11_from_path(wm.pA, wm.pB, lam, path=cfg.path, path_params=cfg.path_params)
            for mk, mv in meta.items():
                row[f"{mk}_w{w}"] = float(mv)
            # Construct full joint
            cells = joint_cells_from_marginals(wm.pA, wm.pB, p11)
            row[f"p00_true_w{w}"] = float(cells["p00"])
            row[f"p01_true_w{w}"] = float(cells["p01"])
            row[f"p10_true_w{w}"] = float(cells["p10"])
            row[f"p11_true_w{w}"] = float(cells["p11"])
            # Path-consistent population overlays (derived from the constructed joint)
            pC_true = pC_from_joint(cfg.rule, cells, pA=wm.pA, pB=wm.pB)
            row[f"pC_true_w{w}"] = float(pC_true)
            row[f"phi_true_w{w}"] = float(phi_from_joint(wm.pA, wm.pB, cells["p11"]))
            row[f"tau_true_w{w}"] = float(kendall_tau_a_from_joint(cells))
        # Derived population metrics across worlds
        pC0_true = row.get("pC_true_w0", float("nan"))
        pC1_true = row.get("pC_true_w1", float("nan"))
        dC_pop = pC1_true - pC0_true
        JC_pop = abs(dC_pop)
        JA_pop = abs(cfg.marginals.w1.pA - cfg.marginals.w0.pA)
        JB_pop = abs(cfg.marginals.w1.pB - cfg.marginals.w0.pB)
        Jbest_pop = max(JA_pop, JB_pop)
        CC_pop = (JC_pop / Jbest_pop) if Jbest_pop > 0 else float("nan")
        row["dC_pop"] = float(dC_pop)
        row["JC_pop"] = float(JC_pop)
        row["JA_pop"] = float(JA_pop)
        row["JB_pop"] = float(JB_pop)
        row["Jbest_pop"] = float(Jbest_pop)
        row["CC_pop"] = float(CC_pop)
        phi0_true = row.get("phi_true_w0", float("nan"))
        phi1_true = row.get("phi_true_w1", float("nan"))
        tau0_true = row.get("tau_true_w0", float("nan"))
        tau1_true = row.get("tau_true_w1", float("nan"))
        row["phi_pop_avg"] = float(0.5 * (phi0_true + phi1_true))
        row["tau_pop_avg"] = float(0.5 * (tau0_true + tau1_true))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("lambda").reset_index(drop=True)
def estimate_thresholds(
    df_pop: pd.DataFrame,
    df_sum: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Estimate population and empirical thresholds from curves.
    """
    out: Dict[str, Any] = {}
    # Population threshold
    out["lambda_star_pop"] = _interp_root(
        df_pop["lambda"].to_numpy(dtype=float),
        df_pop["CC_pop"].to_numpy(dtype=float),
        1.0,
    )
    out["phi_star_pop"] = _map_at_lambda(df_pop, out["lambda_star_pop"], "phi_pop_avg")
    out["tau_star_pop"] = _map_at_lambda(df_pop, out["lambda_star_pop"], "tau_pop_avg")
    # Empirical threshold (if simulation ran)
    if not df_sum.empty:
        out["lambda_star_emp"] = _interp_root(
            df_sum["lambda"].to_numpy(dtype=float),
            df_sum["CC_hat_mean"].to_numpy(dtype=float),
            1.0,
        )
        out["phi_star_emp"] = _map_at_lambda(df_sum, out["lambda_star_emp"], "phi_hat_avg_mean")
        out["tau_star_emp"] = _map_at_lambda(df_sum, out["lambda_star_emp"], "tau_hat_avg_mean")
    return out
def write_manifest(
    *,
    out_dir: Path,
    cfg_dict: Dict[str, Any],
    cfg_resolved: Dict[str, Any],
    thresholds: Dict[str, Any],
    figure_paths: Dict[str, str],
    files_hashed: Dict[str, str],
    run_meta: Dict[str, Any],
) -> None:
    """
    Write a reproducibility manifest capturing config + environment + outputs.
    """
    repo_root = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path(__file__).resolve().parent
    manifest: Dict[str, Any] = {
        "run_started_utc": run_meta.get("run_started_utc"),
        "run_finished_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": float(run_meta.get("elapsed_seconds", float("nan"))),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": getattr(np, "__version__", None),
        "pandas": getattr(pd, "__version__", None),
        "git_commit": _maybe_git_commit(repo_root),
        "config_raw": cfg_dict,
        "config_resolved": cfg_resolved,
        "thresholds": thresholds,
        "figures": figure_paths,
        "file_sha256": files_hashed,
    }
    _write_json(out_dir / "manifest.json", manifest)
# ----------------------------
# Main runner
# ----------------------------
def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    YAML loader (explicit dependency).
    """
    try:
        import yaml # type: ignore
    except Exception as e:
        raise ImportError("pyyaml is required to run run_all.py with a YAML config") from e
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if obj is None:
        return {}
    if not isinstance(obj, Mapping):
        raise ValueError("Config YAML must parse to a mapping/dict.")
    return dict(obj)


def _build_sim_config(cfg: Mapping[str, Any], marg: TwoWorldMarginals, run: RunSpec, lambdas: List[float]) -> SimConfig:
    """
    Build SimConfig from either:
    - simulate.* (preferred)
    - sampling.* + sanity.* (pipeline schema)
    - legacy direct keys
    """
    # Prefer simulate block if present
    sim_blk = cfg.get("simulate", None)
    if isinstance(sim_blk, Mapping):
        n = int(sim_blk.get("n", sim_blk.get("n_per_world", 0)))
        n_reps = int(sim_blk.get("n_reps", 1))
        seed = int(sim_blk.get("seed", 0))
        seed_policy = str(sim_blk.get("seed_policy", "stable_per_cell"))
        envelope_tol = float(sim_blk.get("envelope_tol", 5e-3))
        hard_fail = bool(sim_blk.get("hard_fail_on_invalid", True))
        include_theory_reference = bool(sim_blk.get("include_theory_reference", True))
    else:
        # pipeline or legacy
        n = int(_get(cfg, "sampling.n_per_world", cfg.get("n", 0)))
        n_reps = int(_get(cfg, "sampling.n_reps", cfg.get("n_reps", 1)))
        seed = int(_get(cfg, "sampling.seed", cfg.get("seed", 0)))
        seed_policy = str(_get(cfg, "sampling.seed_policy", _get(cfg, "seed_policy", "stable_per_cell")))
        envelope_tol = float(_get(cfg, "simulate.envelope_tol", _get(cfg, "envelope_tol", 5e-3)))
        hard_fail = bool(_get(cfg, "simulate.hard_fail_on_invalid", _get(cfg, "hard_fail_on_invalid", True)))
        include_theory_reference = bool(_get(cfg, "simulate.include_theory_reference", _get(cfg, "include_theory_reference", True)))

    if n <= 0:
        raise ValueError("Sample size n_per_world (n) must be positive.")
    if n_reps <= 0:
        raise ValueError("n_reps must be positive.")

    # extra numerical policy from sanity or simulate block (optional)
    prob_tol = float(_get(cfg, "simulate.prob_tol", 1e-12))
    allow_tiny_negative = bool(_get(cfg, "simulate.allow_tiny_negative", True))
    tiny_negative_eps = float(_get(cfg, "simulate.tiny_negative_eps", 1e-15))

    return SimConfig(
        marginals=marg,
        rule=run.rule, # type: ignore
        lambdas=lambdas,
        n=n,
        n_reps=n_reps,
        seed=seed,
        path=run.path, # type: ignore
        path_params=dict(run.path_params),
        seed_policy=seed_policy, # type: ignore
        envelope_tol=envelope_tol,
        hard_fail_on_invalid=hard_fail,
        prob_tol=prob_tol,
        allow_tiny_negative=allow_tiny_negative,
        tiny_negative_eps=tiny_negative_eps,
        include_theory_reference=include_theory_reference,
    )


def _maybe_refine_grid(
    *,
    lambdas_coarse: List[float],
    lam_star: Optional[float],
    enabled: bool,
    half_width: float,
    num: int,
) -> List[float]:
    if not enabled or lam_star is None or not math.isfinite(lam_star):
        return sorted({float(x) for x in lambdas_coarse})
    a = max(0.0, float(lam_star) - float(half_width))
    b = min(1.0, float(lam_star) + float(half_width))
    fine = _linspace_grid(a, b, int(num))
    return sorted({float(x) for x in (list(lambdas_coarse) + fine)})


def run(config_path: Path, out_dir: Optional[Path] = None, *, skip_sim: bool = False, skip_figures: bool = False) -> Path:
    """
    Execute the full pipeline.
    Returns the output directory used.
    """
    t0 = time.time()
    run_started_utc = datetime.utcnow().isoformat() + "Z"

    cfg_dict = _load_yaml(config_path)
    cfg_map: Mapping[str, Any] = cfg_dict

    # Resolve runs from config (enterprise matrix or legacy single run)
    marg, runs, resolved_meta = _resolve_runs(cfg_map)

    # Output directory base
    base_out, out_policy = _resolve_output_base(cfg_map, config_path)
    if out_dir is None:
        out_dir = base_out
    # Enforce overwrite policy
    if out_dir.exists() and any(out_dir.iterdir()) and not out_policy.get("overwrite", False):
        out_dir = out_dir / f"run_{_now_stamp()}"
    _ensure_dir(out_dir)

    # Execution options
    dependence_x = str(_get(cfg_map, "reporting.dependence_x", cfg_map.get("dependence_x", "lambda")))
    title_prefix = str(_get(cfg_map, "experiment.name", _get(cfg_map, "title_prefix", "")))
    neutrality_eta = float(_get(cfg_map, "bootstrap.threshold.neutrality_eta", _get(cfg_map, "reporting.neutrality_band_eta", _get(cfg_map, "neutrality_eta", 0.05))))
    figure_formats = _get(cfg_map, "reporting.figure_formats", None)
    also_png = True
    if isinstance(figure_formats, list):
        also_png = "png" in [str(x).lower() for x in figure_formats]
    else:
        also_png = bool(_get(cfg_map, "also_png", True))

    files_hashed: Dict[str, str] = {}
    run_index: List[Dict[str, Any]] = []

    # Run each (rule × path) combo into its own subdir for hygiene
    for rs in runs:
        subdir = out_dir / f"rule_{_safe_subdir_name(rs.rule)}" / f"path_{_safe_subdir_name(rs.path)}"
        _ensure_dir(subdir)

        LOG.info("Running: rule=%s path=%s out=%s", rs.rule, rs.path, subdir)

        # 1) Coarse population curve (for refinement decision)
        sim_cfg_coarse = _build_sim_config(cfg_map, marg, rs, lambdas=rs.lambdas_coarse)
        df_pop_coarse = population_curve_from_path(sim_cfg_coarse)
        df_pop_coarse.to_csv(subdir / "population_curve_coarse.csv", index=False)

        # Estimate coarse population threshold to drive refinement
        lam_star_pop_coarse = _interp_root(
            df_pop_coarse["lambda"].to_numpy(dtype=float),
            df_pop_coarse["CC_pop"].to_numpy(dtype=float),
            1.0,
        )

        lambdas_final = _maybe_refine_grid(
            lambdas_coarse=rs.lambdas_coarse,
            lam_star=lam_star_pop_coarse,
            enabled=rs.refine_enabled,
            half_width=rs.refine_half_width,
            num=rs.refine_num,
        )

        # 2) Final population curve (possibly refined)
        sim_cfg = _build_sim_config(cfg_map, marg, rs, lambdas=lambdas_final)
        df_pop = population_curve_from_path(sim_cfg)
        df_pop.to_csv(subdir / "population_curve.csv", index=False)

        # 3) Simulation
        if not skip_sim:
            df_long = simulate_grid(sim_cfg)
            df_long.to_csv(subdir / "sim_long.csv", index=False)
            df_sum = summarize_simulation(df_long)
            df_sum.to_csv(subdir / "sim_summary.csv", index=False)
        else:
            df_long = pd.DataFrame()
            df_sum = pd.DataFrame()

        # 4) Thresholds
        thresholds = estimate_thresholds(df_pop, df_sum)
        _write_json(subdir / "thresholds.json", thresholds)

        # 5) Figures
        figure_paths: Dict[str, str] = {}
        if not skip_figures:
            style = FigureStyle(title_prefix=title_prefix, neutrality_eta=neutrality_eta)
            figure_paths = make_all_figures(
                out_dir=subdir,
                df_pop=df_pop,
                df_sum=df_sum,
                thresholds=thresholds,
                dependence_x=dependence_x,
                style=style,
                also_png=also_png,
            )

        # Hash outputs (optional)
        if out_policy.get("save_hashes", False):
            for p in (subdir / "population_curve.csv", subdir / "thresholds.json"):
                if p.exists():
                    files_hashed[str(p.relative_to(out_dir))] = _sha256_file(p)
            for p in (subdir / "sim_long.csv", subdir / "sim_summary.csv"):
                if p.exists():
                    files_hashed[str(p.relative_to(out_dir))] = _sha256_file(p)
            for _, fp in figure_paths.items():
                pp = Path(fp)
                if pp.exists():
                    try:
                        files_hashed[str(pp.relative_to(out_dir))] = _sha256_file(pp)
                    except Exception:
                        pass

        run_index.append(
            {
                "rule": rs.rule,
                "path": rs.path,
                "out_dir": str(subdir),
                "refined": bool(rs.refine_enabled),
                "lambda_star_pop": thresholds.get("lambda_star_pop"),
                "lambda_star_emp": thresholds.get("lambda_star_emp"),
            }
        )

    # Save resolved config snapshot (normalized + resolved meta)
    cfg_resolved = {
        "meta": {
            "run_started_utc": run_started_utc,
            "schema_detected": resolved_meta.get("schema_detected"),
            "resolved": resolved_meta,
            "output_base": str(out_dir),
            "output_policy": out_policy,
        },
        "config_raw_path": str(config_path),
        "config_raw_sha256": _sha256_file(config_path) if config_path.exists() else None,
        "marginals": {
            "w0": {"pA": float(marg.w0.pA), "pB": float(marg.w0.pB)},
            "w1": {"pA": float(marg.w1.pA), "pB": float(marg.w1.pB)},
        },
        "runs": run_index,
        "rendering": {
            "dependence_x": dependence_x,
            "title_prefix": title_prefix,
            "neutrality_eta": neutrality_eta,
            "also_png": also_png,
        },
    }
    _write_json(out_dir / "config_resolved.json", cfg_resolved)

    # Top-level index for multi-run batches
    _write_json(out_dir / "index.json", {"runs": run_index})

    # Manifest
    run_meta = {"run_started_utc": run_started_utc, "elapsed_seconds": float(time.time() - t0)}
    # Use last thresholds/figures for manifest fields; index.json is canonical for multi-run.
    write_manifest(
        out_dir=out_dir,
        cfg_dict=cfg_dict,
        cfg_resolved=cfg_resolved,
        thresholds={"runs": run_index},
        figure_paths={},
        files_hashed=files_hashed,
        run_meta=run_meta,
    )
    return out_dir
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Run the full correlation_cliff experiment pipeline.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config (default: config_s1.yaml next to this file).")
    ap.add_argument("--out_dir", type=str, default=None, help="Override output directory.")
    ap.add_argument("--skip_sim", action="store_true", help="Skip Monte Carlo simulation; only compute population curves + thresholds.")
    ap.add_argument("--skip_figures", action="store_true", help="Skip figure rendering.")
    ap.add_argument("--log_level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    args = ap.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config_path = Path(args.config) if args.config else (Path(__file__).resolve().parent / "config_s1.yaml")
    out_dir = Path(args.out_dir) if args.out_dir else None
    used_out = run(config_path=config_path, out_dir=out_dir, skip_sim=bool(args.skip_sim), skip_figures=bool(args.skip_figures))
    print(f"[correlation_cliff] done. outputs written to: {used_out}")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())