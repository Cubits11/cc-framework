import argparse
import sys

from cc.cartographer import atlas, audit, bounds, io, stats


def _cmd_run(argv):
    p = argparse.ArgumentParser(prog="cc.cartographer.cli run")
    p.add_argument("--config", required=True)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--fig", required=True)
    p.add_argument("--audit", required=True)
    p.add_argument("--post-comment", type=str, default="false")
    p.add_argument("--github-context", type=str, default="")
    args = p.parse_args(argv)

    cfg = io.load_config(args.config)
    data = io.load_scores(cfg, n=args.samples)

    JA, JA_ci = stats.compute_j_ci(data["A0"], data["A1"])
    JB, JB_ci = stats.compute_j_ci(data["B0"], data["B1"])

    if data.get("Comp0") is not None:
        Jc, Jc_ci = stats.compute_j_ci(data["Comp0"], data["Comp1"])
        comp_label = "empirical"
    else:
        Jc = bounds.frechet_upper(data["rocA"], data["rocB"], comp=cfg.get("comp", "AND"))
        Jc_ci = (None, None)
        comp_label = "UPPER BOUND"

    CC, Dadd = stats.compose_cc(JA, JB, Jc)
    fig_path = atlas.plot_phase_point(cfg, CC, args.fig)
    entry, decision = atlas.compose_entry(
        cfg, JA, JA_ci, JB, JB_ci, Jc, Jc_ci, CC, Dadd, comp_label, fig_path
    )

    rec = audit.make_record(cfg, JA, JA_ci, JB, JB_ci, Jc, Jc_ci, CC, Dadd, decision, [fig_path])
    sha = audit.append_jsonl(args.audit, rec)

    print(entry)
    print(decision)
    print("audit_sha:", sha)

    if args.post_comment.lower() == "true" and args.github_context:
        io.post_github_comment(
            args.github_context, entry + "\n\n" + decision + f"\n\n**audit_sha:** `{sha}`"
        )


def _cmd_verify_audit(argv):
    p = argparse.ArgumentParser(prog="cc.cartographer.cli verify-audit")
    p.add_argument("--audit", required=True)
    args = p.parse_args(argv)
    audit.verify_chain(args.audit)
    print("audit chain OK")


def _cmd_verify_stats(argv):
    p = argparse.ArgumentParser(prog="cc.cartographer.cli verify-stats")
    p.add_argument("--config", required=True)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args(argv)
    cfg = io.load_config(args.config)
    _ = io.load_scores(cfg, n=None)
    stats.bootstrap_diagnostics(_, B=args.bootstrap)
    print("bootstrap diagnostics OK")


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python -m cc.cartographer.cli {run|verify-audit|verify-stats} ...",
            file=sys.stderr,
        )
        sys.exit(2)

    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        _cmd_run(argv)
    elif cmd == "verify-audit":
        _cmd_verify_audit(argv)
    elif cmd == "verify-stats":
        _cmd_verify_stats(argv)
    else:
        print(f"unknown subcommand: {cmd}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
