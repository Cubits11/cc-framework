from __future__ import annotations

"""
simulate.cli
============

CLI entrypoint + YAML schema parsing.

This file owns:
- YAML load errors (pyyaml optional, required only for --config)
- legacy + pipeline schema mapping -> SimConfig
- artifact bundle writing + stable JSON diagnostics
"""

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path as SysPath
from typing import Any

import numpy as np
import pandas as pd

from . import utils as U
from .config import SimConfig, validate_cfg
from .core import simulate_grid, summarize_simulation

LOG = logging.getLogger(__name__)


def _load_yaml(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ImportError("pyyaml is required for --config usage (pip install pyyaml).") from e

    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        raise OSError(f"Could not read YAML config at path={path!r}: {e}") from e

    try:
        obj = yaml.safe_load(raw)
    except Exception as e:
        mark = getattr(e, "problem_mark", None)
        if mark is not None:
            loc = f"line={getattr(mark, 'line', '?')}, column={getattr(mark, 'column', '?')}"
            raise ValueError(f"YAML parse error in {path!r} ({loc}): {e}") from e
        raise ValueError(f"YAML parse error in {path!r}: {e}") from e

    if obj is None:
        return {}
    if not isinstance(obj, dict):
        raise ValueError(f"YAML config must parse to a mapping/dict, got {type(obj).__name__}")
    return dict(obj)


def _cfg_from_dict(d: dict[str, Any]) -> SimConfig:
    def _dget(obj: Any, dotted: str, default: Any = None) -> Any:
        cur = obj
        for k in dotted.split("."):
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    def _require_dict(x: Any, name: str) -> dict[str, Any]:
        if not isinstance(x, dict):
            raise ValueError(f"Expected mapping for '{name}', got {type(x).__name__}")
        return x

    def _f(x: Any, name: str, *, finite: bool = True) -> float:
        try:
            v = float(x)
        except Exception as e:
            raise ValueError(f"{name} must be a number, got {x!r}") from e
        if finite and not np.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
        return v

    def _i(x: Any, name: str) -> int:
        try:
            v = int(x)
        except Exception as e:
            raise ValueError(f"{name} must be an int, got {x!r}") from e
        return v

    def _b(x: Any, name: str) -> bool:
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)) and x in (0, 1):
            return bool(x)
        if isinstance(x, str):
            s = x.strip().lower()
            if s in ("true", "yes", "y", "1"):
                return True
            if s in ("false", "no", "n", "0"):
                return False
        raise ValueError(f"{name} must be a bool, got {x!r}")

    def _p(x: Any, name: str) -> float:
        v = _f(x, name, finite=True)
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{name} must be in [0,1], got {v}")
        return v

    def _parse_path_params(base: Any, *, extra: dict[str, Any]) -> dict[str, Any]:
        pp: dict[str, Any] = {}
        if isinstance(base, dict):
            pp.update(dict(base))
        for k, v in extra.items():
            if v is not None and k not in pp:
                pp[k] = v
        return pp

    def _parse_lambdas_from_grid(grid_obj: Any, *, where: str):
        g = _require_dict(grid_obj, where)
        start = _f(g.get("start", 0.0), f"{where}.start")
        stop = _f(g.get("stop", 1.0), f"{where}.stop")
        num = _i(g.get("num", 21), f"{where}.num")
        arr = U.build_linear_lambda_grid(num=num, start=start, stop=stop, closed="both")
        return [float(x) for x in arr.tolist()]

    def _ensure_monotone_increasing(lams):
        if len(lams) == 0:
            raise ValueError("lambdas must be non-empty.")
        xs = [float(x) for x in lams]
        for i, x in enumerate(xs):
            if not (np.isfinite(x) and 0.0 <= x <= 1.0):
                raise ValueError(
                    f"lambda values must be finite and in [0,1]; bad lambdas[{i}]={x!r}"
                )
        for i in range(len(xs) - 1):
            if xs[i + 1] < xs[i]:
                raise ValueError(
                    f"lambdas must be non-decreasing; found lambdas[{i}]={xs[i]} > lambdas[{i + 1}]={xs[i + 1]}"
                )
        return xs

    md = _require_dict(d.get("marginals", {}), "marginals")
    w0 = _require_dict(md.get("w0", {}), "marginals.w0")
    w1 = _require_dict(md.get("w1", {}), "marginals.w1")
    if "pA" not in w0 or "pB" not in w0 or "pA" not in w1 or "pB" not in w1:
        raise ValueError("marginals.w0 and marginals.w1 must each define pA and pB.")

    marg = U.TwoWorldMarginals(
        w0=U.WorldMarginals(pA=_p(w0["pA"], "marginals.w0.pA"), pB=_p(w0["pB"], "marginals.w0.pB")),
        w1=U.WorldMarginals(pA=_p(w1["pA"], "marginals.w1.pA"), pB=_p(w1["pB"], "marginals.w1.pB")),
    )

    has_pipeline = (
        isinstance(d.get("composition"), dict)
        or isinstance(d.get("dependence_paths"), dict)
        or isinstance(d.get("sampling"), dict)
    )

    if has_pipeline:
        rule = str(_dget(d, "composition.primary_rule", "OR")).upper()
        if rule not in ("OR", "AND"):
            raise ValueError(f"composition.primary_rule must be OR/AND, got {rule!r}")

        path = str(_dget(d, "dependence_paths.primary.type", "fh_linear"))
        if path not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
            raise ValueError(f"dependence_paths.primary.type invalid: {path!r}")

        primary_dep = _dget(d, "dependence_paths.primary", {})
        if not isinstance(primary_dep, dict):
            raise ValueError("dependence_paths.primary must be a mapping if provided.")
        extra_params = {
            "gamma": primary_dep.get("gamma", None),
            "k": primary_dep.get("k", None),
            "ppf_clip_eps": primary_dep.get("ppf_clip_eps", None),
        }
        path_params = _parse_path_params(primary_dep.get("path_params", {}), extra=extra_params)

        if "lambdas" in d:
            lambdas = [float(x) for x in (d.get("lambdas") or [])]
        else:
            grid = _dget(d, "dependence_paths.primary.lambda_grid_coarse", None) or _dget(
                d, "dependence_paths.primary.lambda_grid", None
            )
            if grid is None:
                grid = d.get("lambda_grid", {"num": 21})
            lambdas = list(
                _parse_lambdas_from_grid(grid, where="dependence_paths.primary.lambda_grid_coarse")
            )

        lambdas = list(_ensure_monotone_increasing(lambdas))

        n = _i(_dget(d, "sampling.n_per_world", d.get("n")), "sampling.n_per_world")
        n_reps = _i(_dget(d, "sampling.n_reps", d.get("n_reps", 1)), "sampling.n_reps")
        seed = _i(_dget(d, "sampling.seed", d.get("seed", 0)), "sampling.seed")
        seed_policy = str(_dget(d, "sampling.seed_policy", d.get("seed_policy", "stable_per_cell")))
        if seed_policy not in ("stable_per_cell", "sequential"):
            raise ValueError(f"sampling.seed_policy invalid: {seed_policy!r}")

        envelope_tol = _f(
            _dget(d, "simulate.envelope_tol", d.get("envelope_tol", 5e-3)), "simulate.envelope_tol"
        )
        hard_fail_on_invalid = _b(
            _dget(d, "simulate.hard_fail_on_invalid", d.get("hard_fail_on_invalid", True)),
            "simulate.hard_fail_on_invalid",
        )
        include_theory_reference = _b(
            _dget(d, "simulate.include_theory_reference", d.get("include_theory_reference", True)),
            "simulate.include_theory_reference",
        )

        prob_tol = _f(_dget(d, "simulate.prob_tol", d.get("prob_tol", 1e-12)), "simulate.prob_tol")
        allow_tiny_negative = _b(
            _dget(d, "simulate.allow_tiny_negative", d.get("allow_tiny_negative", True)),
            "simulate.allow_tiny_negative",
        )
        tiny_negative_eps = _f(
            _dget(d, "simulate.tiny_negative_eps", d.get("tiny_negative_eps", 1e-15)),
            "simulate.tiny_negative_eps",
        )

    else:
        rule = str(d.get("rule", "OR")).upper()
        if rule not in ("OR", "AND"):
            raise ValueError(f"Invalid rule: {rule!r}")

        path = str(d.get("path", "fh_linear"))
        if path not in ("fh_linear", "fh_power", "fh_scurve", "gaussian_tau"):
            raise ValueError(f"Invalid path: {path!r}")

        base_pp = d.get("path_params", {})
        extra_pp = {
            "gamma": d.get("gamma"),
            "k": d.get("k"),
            "ppf_clip_eps": d.get("ppf_clip_eps"),
        }
        path_params = _parse_path_params(base_pp, extra=extra_pp)

        if "lambdas" in d:
            lambdas = [float(x) for x in (d.get("lambdas") or [])]
        else:
            lg = d.get("lambda_grid", {"num": 21})
            lambdas = list(_parse_lambdas_from_grid(lg, where="lambda_grid"))
        lambdas = list(_ensure_monotone_increasing(lambdas))

        if "n" not in d:
            raise ValueError("Legacy schema requires top-level 'n' (sample size per world).")
        n = _i(d["n"], "n")
        n_reps = _i(d.get("n_reps", 1), "n_reps")
        seed = _i(d.get("seed", 0), "seed")

        seed_policy = str(d.get("seed_policy", "stable_per_cell"))
        if seed_policy not in ("stable_per_cell", "sequential"):
            raise ValueError(f"Invalid seed_policy: {seed_policy!r}")

        envelope_tol = _f(d.get("envelope_tol", 5e-3), "envelope_tol")
        hard_fail_on_invalid = bool(d.get("hard_fail_on_invalid", True))
        include_theory_reference = bool(d.get("include_theory_reference", True))

        prob_tol = _f(d.get("prob_tol", 1e-12), "prob_tol")
        allow_tiny_negative = bool(d.get("allow_tiny_negative", True))
        tiny_negative_eps = _f(d.get("tiny_negative_eps", 1e-15), "tiny_negative_eps")

    if n <= 0:
        raise ValueError("n must be positive.")
    if n_reps <= 0:
        raise ValueError("n_reps must be positive.")

    return SimConfig(
        marginals=marg,
        rule=rule,  # type: ignore
        lambdas=list(lambdas),
        n=int(n),
        n_reps=int(n_reps),
        seed=int(seed),
        path=path,  # type: ignore
        path_params=dict(path_params),
        seed_policy=seed_policy,  # type: ignore
        envelope_tol=float(envelope_tol),
        hard_fail_on_invalid=bool(hard_fail_on_invalid),
        prob_tol=float(prob_tol),
        allow_tiny_negative=bool(allow_tiny_negative),
        tiny_negative_eps=float(tiny_negative_eps),
        include_theory_reference=bool(include_theory_reference),
    )


def main(argv: Sequence[str] | None = None) -> int:
    LOG = logging.getLogger("correlation_cliff.simulate")

    def _now_stamp() -> str:
        return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    def _ensure_dir(p: SysPath) -> None:
        p.mkdir(parents=True, exist_ok=True)

    def _atomic_write_text(path: SysPath, text: str) -> None:
        _ensure_dir(path.parent)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        tmp.replace(path)

    def _atomic_write_csv(df: pd.DataFrame, path: SysPath) -> None:
        _ensure_dir(path.parent)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)

    def _sha256_file(path: SysPath) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()

    def _maybe_git_commit(repo_root: SysPath) -> str | None:
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

    def _repo_root_guess() -> SysPath:
        here = SysPath(__file__).resolve()
        parents = list(here.parents)
        if len(parents) >= 4:
            # .../experiments/correlation_cliff/simulate/cli.py -> repo root likely 3 up
            return parents[3]
        return here.parent

    ap = argparse.ArgumentParser(description="Run correlation cliff simulation grid.")
    ap.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    ap.add_argument(
        "--out_dir", type=str, default=None, help="Write a full artifact bundle to this directory."
    )
    ap.add_argument(
        "--out_csv", type=str, default=None, help="Write replicate-level rows to CSV (legacy)."
    )
    ap.add_argument(
        "--out_summary_csv",
        type=str,
        default=None,
        help="Write per-lambda summary to CSV (legacy).",
    )
    ap.add_argument("--print_head", type=int, default=5, help="Print first N rows of summary.")
    ap.add_argument("--log_level", type=str, default="INFO", help="Logging level.")
    ap.add_argument(
        "--overwrite", action="store_true", help="Overwrite outputs in out_dir if non-empty."
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    if args.config is None:
        print(
            "ERROR: Please provide --config path/to/config.yaml (or call simulate_grid() from Python).",
            file=sys.stderr,
        )
        return 2

    run_started_utc = datetime.utcnow().isoformat() + "Z"
    t0 = time.time()

    config_path = SysPath(args.config).expanduser()
    if not config_path.exists():
        print(f"ERROR: config path does not exist: {str(config_path)!r}", file=sys.stderr)
        return 2

    out_dir: SysPath | None = SysPath(args.out_dir).expanduser() if args.out_dir else None
    legacy_out_csv: SysPath | None = SysPath(args.out_csv).expanduser() if args.out_csv else None
    legacy_out_sum: SysPath | None = (
        SysPath(args.out_summary_csv).expanduser() if args.out_summary_csv else None
    )

    if out_dir is None and legacy_out_csv is None and legacy_out_sum is None:
        out_dir = SysPath.cwd() / "artifacts" / _now_stamp()

    if out_dir is not None:
        if out_dir.exists() and any(out_dir.iterdir()) and not bool(args.overwrite):
            out_dir = out_dir / f"run_{_now_stamp()}"
        _ensure_dir(out_dir)

    try:
        cfg_dict = _load_yaml(str(config_path))
        cfg = _cfg_from_dict(cfg_dict)
        validate_cfg(cfg)
    except Exception as e:
        LOG.exception("Config loading/validation failed.")
        print(f"ERROR: config loading/validation failed: {e}", file=sys.stderr)
        return 2

    try:
        LOG.info(
            "Running simulate_grid: n=%s n_reps=%s lambdas=%s path=%s rule=%s seed_policy=%s",
            cfg.n,
            cfg.n_reps,
            len(cfg.lambdas),
            cfg.path,
            cfg.rule,
            cfg.seed_policy,
        )
        df_long = simulate_grid(cfg)
        df_sum = summarize_simulation(df_long)
    except Exception as e:
        LOG.exception("Simulation failed.")
        print(f"ERROR: simulation failed: {e}", file=sys.stderr)
        return 1

    file_hashes: dict[str, str] = {}
    outputs: dict[str, str] = {}

    try:
        if out_dir is not None:
            long_path = out_dir / "sim_long.csv"
            sum_path = out_dir / "sim_summary.csv"
            _atomic_write_csv(df_long, long_path)
            _atomic_write_csv(df_sum, sum_path)
            outputs["sim_long.csv"] = str(long_path)
            outputs["sim_summary.csv"] = str(sum_path)
            file_hashes["sim_long.csv"] = _sha256_file(long_path)
            file_hashes["sim_summary.csv"] = _sha256_file(sum_path)

            cfg_snapshot = {
                "config_path": str(config_path),
                "marginals": {
                    "w0": {"pA": float(cfg.marginals.w0.pA), "pB": float(cfg.marginals.w0.pB)},
                    "w1": {"pA": float(cfg.marginals.w1.pA), "pB": float(cfg.marginals.w1.pB)},
                },
                "rule": str(cfg.rule),
                "path": str(cfg.path),
                "path_params": dict(cfg.path_params),
                "seed_policy": str(cfg.seed_policy),
                "seed": int(cfg.seed),
                "n": int(cfg.n),
                "n_reps": int(cfg.n_reps),
                "lambda_points": len(cfg.lambdas),
                "lambdas": [float(x) for x in cfg.lambdas],
                "envelope_tol": float(cfg.envelope_tol),
                "hard_fail_on_invalid": bool(cfg.hard_fail_on_invalid),
                "prob_tol": float(cfg.prob_tol),
                "allow_tiny_negative": bool(cfg.allow_tiny_negative),
                "tiny_negative_eps": float(cfg.tiny_negative_eps),
                "include_theory_reference": bool(cfg.include_theory_reference),
            }
            _atomic_write_text(
                out_dir / "config_resolved.json",
                json.dumps(cfg_snapshot, indent=2, sort_keys=True) + "\n",
            )
            outputs["config_resolved.json"] = str(out_dir / "config_resolved.json")
            file_hashes["config_resolved.json"] = _sha256_file(out_dir / "config_resolved.json")

        if legacy_out_csv is not None:
            _ensure_dir(legacy_out_csv.parent)
            _atomic_write_csv(df_long, legacy_out_csv)
            outputs["out_csv"] = str(legacy_out_csv)
        if legacy_out_sum is not None:
            _ensure_dir(legacy_out_sum.parent)
            _atomic_write_csv(df_sum, legacy_out_sum)
            outputs["out_summary_csv"] = str(legacy_out_sum)

    except Exception as e:
        LOG.exception("Writing outputs failed.")
        print(f"ERROR: writing outputs failed: {e}", file=sys.stderr)
        return 1

    vio_rate = (
        float(df_long["JC_env_violation"].mean())
        if "JC_env_violation" in df_long.columns and not df_long.empty
        else float("nan")
    )
    invalid_cols = [c for c in df_long.columns if c.startswith("invalid_joint_w")]
    invalid_rates = {c: float(df_long[c].fillna(False).astype(bool).mean()) for c in invalid_cols}

    diag = {
        "run_started_utc": run_started_utc,
        "run_finished_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": float(time.time() - t0),
        "rows": len(df_long),
        "lambda_points": len(set(df_long["lambda"])) if "lambda" in df_long.columns else 0,
        "env_violation_rate": vio_rate,
        "seed_policy": str(cfg.seed_policy),
        "seed": int(cfg.seed),
        "path": str(cfg.path),
        "rule": str(cfg.rule),
        **{f"{k}_rate": v for k, v in invalid_rates.items()},
    }

    print(json.dumps(diag, sort_keys=True))

    if out_dir is not None:
        try:
            _atomic_write_text(
                out_dir / "diagnostics.json", json.dumps(diag, indent=2, sort_keys=True) + "\n"
            )
            outputs["diagnostics.json"] = str(out_dir / "diagnostics.json")
            file_hashes["diagnostics.json"] = _sha256_file(out_dir / "diagnostics.json")

            repo_root = _repo_root_guess()
            manifest = {
                "run_started_utc": run_started_utc,
                "run_finished_utc": datetime.utcnow().isoformat() + "Z",
                "elapsed_seconds": float(time.time() - t0),
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": getattr(np, "__version__", None),
                "pandas": getattr(pd, "__version__", None),
                "git_commit": _maybe_git_commit(repo_root),
                "config_path": str(config_path),
                "config_sha256": _sha256_file(config_path),
                "outputs": outputs,
                "file_sha256": file_hashes,
                "diagnostics": diag,
            }
            _atomic_write_text(
                out_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            outputs["manifest.json"] = str(out_dir / "manifest.json")
            file_hashes["manifest.json"] = _sha256_file(out_dir / "manifest.json")
        except Exception:
            LOG.exception("Failed to write diagnostics/manifest bundle.")

    if args.print_head and int(args.print_head) > 0:
        with pd.option_context("display.width", 160, "display.max_columns", 200):
            print(df_sum.head(int(args.print_head)))

    if out_dir is not None:
        print(f"[correlation_cliff.simulate] outputs written to: {out_dir}", file=sys.stderr)

    return 0
