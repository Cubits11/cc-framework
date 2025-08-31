from typing import Tuple, List
import matplotlib.pyplot as plt


def plot_phase_point(cfg: dict, CCmax: float, outfile: str) -> str:
    plt.figure()
    plt.scatter([cfg.get("epsilon", 0)], [cfg.get("T", 0)], s=80)
    plt.title(f"CC phase point: CC_max={CCmax:.2f}")
    plt.xlabel("epsilon")
    plt.ylabel("T")
    plt.savefig(outfile, bbox_inches="tight")
    plt.close()
    return outfile


def compose_entry(cfg, JA, JA_ci, JB, JB_ci, Jc, Jc_ci, CC, Dadd, comp_label: str, fig_path: str):
    eps, T = cfg.get("epsilon", "?"), cfg.get("T", "?")
    comp = cfg.get("comp", "AND")
    entry = (
        f"ε={eps}, T={T}; {cfg.get('A','A')} ⊕ {cfg.get('B','B')} ({comp}) — "
        f"J_A={JA:.2f} [{JA_ci[0]:.2f}, {JA_ci[1]:.2f}], "
        f"J_B={JB:.2f} [{JB_ci[0]:.2f}, {JB_ci[1]:.2f}], "
        f"J_comp({comp_label})={Jc:.2f}"
        f"{'' if Jc_ci[0] is None else f' [{Jc_ci[0]:.2f}, {Jc_ci[1]:.2f}]'} ⇒ "
        f"CC_max={CC:.2f}, Δ_add={Dadd:+.2f}. "
        f"{_region(CC)}. Next: probe nearby (ε,T)."
    )
    decision = f"DECISION: {_decision(CC)} — reason: CC_max={CC:.2f}; refs: {fig_path}"
    return entry, decision


def _region(CC):
    if CC < 0.95:
        return "Constructive Valley"
    if CC <= 1.05:
        return "Independent Plateau"
    return "Red Wedge (Destructive)"


def _decision(CC):
    if CC < 0.95:
        return "ADOPT HYBRID"
    if CC <= 1.05:
        return "PREFER SINGLE"
    return "REDESIGN"
