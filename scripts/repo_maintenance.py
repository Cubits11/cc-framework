#!/usr/bin/env python3
"""Guarded repository maintenance for branch consolidation and cleanup.

The script is dry-run by default. Pass --execute only after reviewing the plan.
It is intentionally conservative:

* updates main from the remote before merging;
* merges only explicitly designated branches, unless --merge-all-side-branches is set;
* runs quality gates before pushing main;
* refuses to delete side branches whose tips are not ancestors of main, unless forced;
* applies GitHub branch protection through gh after the consolidation push.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Sequence

DEFAULT_MAIN = "main"
DEFAULT_REMOTE = "origin"
DEFAULT_QUALITY_COMMANDS = ("ruff check .", "make type", "pytest -q")
DEFAULT_CI_CONTEXTS = (
    "CI / lint-type-test (3.10)",
    "CI / lint-type-test (3.11)",
    "CI / lint-type-test (3.12)",
    "CI / lint-type-test (3.13)",
    "CI / enterprise-smoke",
    "Pre-Commit / pre-commit",
)


class MaintenanceError(RuntimeError):
    """Raised when maintenance cannot proceed safely."""


@dataclass(frozen=True)
class RunResult:
    stdout: str = ""
    returncode: int = 0


class Runner:
    def __init__(self, execute: bool) -> None:
        self.execute = execute

    def run(
        self,
        cmd: Sequence[str] | str,
        *,
        capture: bool = False,
        check: bool = True,
        input_text: str | None = None,
        mutate: bool = True,
        shell: bool = False,
    ) -> RunResult:
        display = cmd if isinstance(cmd, str) else shlex.join(cmd)
        if mutate and not self.execute:
            print(f"DRY-RUN: {display}")
            return RunResult()

        print(f"+ {display}")
        completed = subprocess.run(
            cmd,
            check=False,
            input=input_text,
            shell=shell,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            text=True,
        )
        stdout = completed.stdout or ""
        if capture and stdout:
            print(stdout, end="")
        if check and completed.returncode != 0:
            raise MaintenanceError(f"Command failed ({completed.returncode}): {display}")
        return RunResult(stdout=stdout, returncode=completed.returncode)


def split_branch_names(*values: str | None) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for item in re.split(r"[\s,]+", value.strip()):
            if item and item not in seen:
                items.append(item)
                seen.add(item)
    return items


def split_records(*values: str | None) -> list[str]:
    records: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        for record in re.split(r"[\n,]+", value):
            item = record.strip()
            if item and item not in seen:
                records.append(item)
                seen.add(item)
    return records


def git(runner: Runner, args: Sequence[str], **kwargs: object) -> RunResult:
    return runner.run(["git", *args], **kwargs)


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise MaintenanceError(f"Required tool not found on PATH: {name}")


def ensure_clean_worktree(runner: Runner, allow_dirty: bool) -> None:
    status = git(runner, ["status", "--porcelain"], capture=True, mutate=False).stdout.strip()
    if status and not allow_dirty:
        raise MaintenanceError(
            "Working tree is not clean. Commit/stash changes first, or pass --allow-dirty."
        )


def ref_exists(runner: Runner, ref: str) -> bool:
    result = git(
        runner,
        ["rev-parse", "--verify", "--quiet", ref],
        capture=True,
        check=False,
        mutate=False,
    )
    return result.returncode == 0


def local_branches(runner: Runner) -> list[str]:
    output = git(
        runner,
        ["for-each-ref", "refs/heads", "--format=%(refname:short)"],
        capture=True,
        mutate=False,
    ).stdout
    return [line.strip() for line in output.splitlines() if line.strip()]


def remote_branches(runner: Runner, remote: str) -> list[str]:
    output = git(
        runner,
        ["for-each-ref", f"refs/remotes/{remote}", "--format=%(refname:short)"],
        capture=True,
        mutate=False,
    ).stdout
    prefix = f"{remote}/"
    branches: list[str] = []
    for line in output.splitlines():
        ref = line.strip()
        if not ref or ref == f"{remote}/HEAD" or not ref.startswith(prefix):
            continue
        branches.append(ref[len(prefix) :])
    return branches


def side_branches(runner: Runner, remote: str, main: str) -> list[str]:
    names = {*local_branches(runner), *remote_branches(runner, remote)}
    return sorted(name for name in names if name != main)


def resolve_branch_refs(runner: Runner, remote: str, branch: str) -> list[str]:
    remote_prefix = f"{remote}/"
    if branch.startswith(remote_prefix) and ref_exists(runner, branch):
        return [branch]

    remote_ref = f"{remote}/{branch}"
    has_remote = ref_exists(runner, remote_ref)
    has_local = ref_exists(runner, branch)

    if has_remote and has_local:
        if is_ancestor(runner, branch, remote_ref):
            return [remote_ref]
        if is_ancestor(runner, remote_ref, branch):
            return [branch]
        return [remote_ref, branch]

    if has_remote:
        return [remote_ref]
    if has_local:
        return [branch]

    raise MaintenanceError(f"Branch not found locally or on {remote}: {branch}")


def is_ancestor(runner: Runner, ancestor_ref: str, descendant_ref: str) -> bool:
    result = git(
        runner,
        ["merge-base", "--is-ancestor", ancestor_ref, descendant_ref],
        check=False,
        mutate=False,
    )
    return result.returncode == 0


def remote_url(runner: Runner, remote: str) -> str:
    return git(runner, ["remote", "get-url", remote], capture=True, mutate=False).stdout.strip()


def infer_repo_slug(url: str) -> str | None:
    patterns = (
        r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)(?:\.git)?$",
        r"github\.com/(?P<owner>[^/]+)/(?P<repo>[^/.]+)(?:\.git)?$",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return f"{match.group('owner')}/{match.group('repo')}"
    return None


def merge_designated_branches(
    runner: Runner,
    remote: str,
    main_ref: str,
    branches: Sequence[str],
) -> None:
    for branch in branches:
        for ref in resolve_branch_refs(runner, remote, branch):
            if is_ancestor(runner, ref, main_ref):
                print(f"Already merged: {branch} ({ref})")
                continue
            git(runner, ["merge", "--no-ff", "--no-edit", ref], mutate=True)


def quality_gate(runner: Runner, commands: Sequence[str], run_in_dry_run: bool) -> None:
    if not commands:
        raise MaintenanceError("No quality commands configured.")

    print("Running quality gates:")
    for command in commands:
        mutate = not run_in_dry_run
        runner.run(command, shell=True, mutate=mutate)


def unmerged_side_refs(
    runner: Runner, remote: str, main_ref: str, main: str
) -> dict[str, list[str]]:
    unmerged: dict[str, list[str]] = {}
    for branch in side_branches(runner, remote, main):
        refs: list[str] = []
        if ref_exists(runner, branch):
            refs.append(branch)
        remote_ref = f"{remote}/{branch}"
        if ref_exists(runner, remote_ref):
            refs.append(remote_ref)

        for ref in refs:
            if not is_ancestor(runner, ref, main_ref):
                unmerged.setdefault(branch, []).append(ref)
    return unmerged


def delete_side_branches(runner: Runner, remote: str, main: str) -> None:
    for branch in local_branches(runner):
        if branch != main:
            git(runner, ["branch", "-D", branch], mutate=True)

    for branch in remote_branches(runner, remote):
        if branch != main:
            git(runner, ["push", remote, "--delete", branch], mutate=True)


def branch_protection_payload(
    contexts: Sequence[str], required_review_count: int
) -> dict[str, object]:
    reviews: dict[str, object] | None = None
    if required_review_count > 0:
        reviews = {
            "dismiss_stale_reviews": True,
            "require_code_owner_reviews": False,
            "required_approving_review_count": required_review_count,
        }

    return {
        "required_status_checks": (
            {
                "strict": True,
                "contexts": list(contexts),
            }
            if contexts
            else None
        ),
        "enforce_admins": True,
        "required_pull_request_reviews": reviews,
        "restrictions": None,
        "required_linear_history": True,
        "allow_force_pushes": False,
        "allow_deletions": False,
        "required_conversation_resolution": True,
    }


def ensure_branch_protection(
    runner: Runner,
    repo: str,
    main: str,
    contexts: Sequence[str],
    required_review_count: int,
) -> None:
    if runner.execute:
        require_tool("gh")
    payload = json.dumps(branch_protection_payload(contexts, required_review_count), indent=2)
    print(f"Applying branch protection to {repo}:{main}")
    print(payload)
    runner.run(
        [
            "gh",
            "api",
            "--method",
            "PUT",
            f"repos/{repo}/branches/{main}/protection",
            "--header",
            "Accept: application/vnd.github+json",
            "--header",
            "X-GitHub-Api-Version: 2022-11-28",
            "--input",
            "-",
        ],
        input_text=payload,
        mutate=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge designated branches into main, run quality gates, protect main, and delete side branches."
    )
    parser.add_argument(
        "--execute", action="store_true", help="Apply mutations. Default is dry-run."
    )
    parser.add_argument("--main", default=os.getenv("MAIN_BRANCH", DEFAULT_MAIN))
    parser.add_argument("--remote", default=os.getenv("REMOTE", DEFAULT_REMOTE))
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", ""))
    parser.add_argument("--allow-dirty", action="store_true")

    parser.add_argument("--merge-branch", action="append", default=[], dest="merge_branch")
    parser.add_argument(
        "--merge-branches",
        default=os.getenv("PHASE_BRANCHES", ""),
        help="Comma, space, or newline separated branch names to merge before cleanup.",
    )
    parser.add_argument(
        "--merge-all-side-branches",
        action="store_true",
        help="Merge every side branch before cleanup.",
    )
    parser.add_argument(
        "--allow-delete-unmerged",
        action="store_true",
        help="Allow deletion of branches not reachable from main. Dangerous.",
    )

    parser.add_argument("--skip-fetch", action="store_true")
    parser.add_argument("--skip-quality", action="store_true")
    parser.add_argument("--skip-push", action="store_true")
    parser.add_argument("--skip-protection", action="store_true")
    parser.add_argument("--skip-delete", action="store_true")
    parser.add_argument(
        "--run-quality-in-dry-run",
        action="store_true",
        help="Actually run quality commands even without --execute.",
    )
    parser.add_argument(
        "--quality-command",
        action="append",
        default=[],
        help="Quality command to run. Repeatable; overrides defaults when present.",
    )
    parser.add_argument(
        "--quality-commands",
        default=os.getenv("QUALITY_COMMANDS", ""),
        help="Comma or newline separated quality commands. Overrides defaults when set.",
    )

    parser.add_argument("--protection-context", action="append", default=[])
    parser.add_argument(
        "--protection-contexts",
        default=os.getenv("PROTECTION_CONTEXTS", ""),
        help="Comma or newline separated required status check contexts.",
    )
    parser.add_argument(
        "--require-ci-status-checks",
        action="store_true",
        help="Require the repository's default CI and pre-commit contexts on main.",
    )
    parser.add_argument(
        "--required-review-count",
        type=int,
        default=int(os.getenv("REQUIRED_REVIEW_COUNT", "1")),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    runner = Runner(execute=args.execute)

    try:
        require_tool("git")
        ensure_clean_worktree(runner, args.allow_dirty)

        if not args.skip_fetch:
            git(runner, ["fetch", "--all", "--prune"], mutate=True)

        discovered_side_branches = side_branches(runner, args.remote, args.main)
        print(f"Side branches discovered: {', '.join(discovered_side_branches) or '(none)'}")

        merge_branches = split_branch_names(*args.merge_branch, args.merge_branches)
        if args.merge_all_side_branches:
            merge_branches = discovered_side_branches
        print(f"Designated branches to merge: {', '.join(merge_branches) or '(none)'}")

        if args.execute:
            git(runner, ["checkout", args.main], mutate=True)
            git(runner, ["pull", "--ff-only", args.remote, args.main], mutate=True)
            main_ref = "HEAD"
        else:
            main_ref = args.main if ref_exists(runner, args.main) else f"{args.remote}/{args.main}"
            print(f"Dry-run main comparison ref: {main_ref}")

        if merge_branches:
            merge_designated_branches(runner, args.remote, main_ref, merge_branches)

        if not args.skip_quality:
            quality_commands = split_records(args.quality_commands) or args.quality_command
            if not quality_commands:
                quality_commands = list(DEFAULT_QUALITY_COMMANDS)
            quality_gate(runner, quality_commands, args.run_quality_in_dry_run)

        if not args.skip_push:
            git(runner, ["push", args.remote, args.main], mutate=True)

        if not args.skip_protection:
            repo = args.repo or infer_repo_slug(remote_url(runner, args.remote))
            if not repo:
                raise MaintenanceError(
                    "Could not infer GitHub repo slug from remote URL. Pass --repo owner/name."
                )
            contexts = split_records(*args.protection_context, args.protection_contexts)
            if args.require_ci_status_checks:
                contexts = [
                    *DEFAULT_CI_CONTEXTS,
                    *[c for c in contexts if c not in DEFAULT_CI_CONTEXTS],
                ]
            ensure_branch_protection(runner, repo, args.main, contexts, args.required_review_count)

        if not args.skip_delete:
            if args.execute:
                main_ref = "HEAD"
            unmerged = unmerged_side_refs(runner, args.remote, main_ref, args.main)
            if unmerged and not args.allow_delete_unmerged:
                details = "\n".join(
                    f"  - {branch}: {', '.join(refs)}" for branch, refs in sorted(unmerged.items())
                )
                raise MaintenanceError(
                    "Refusing to delete unmerged side branches:\n"
                    f"{details}\n"
                    "Merge them first or pass --allow-delete-unmerged."
                )
            delete_side_branches(runner, args.remote, args.main)

    except MaintenanceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Repository maintenance completed." if args.execute else "Dry-run completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
