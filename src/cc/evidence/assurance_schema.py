"""GSN-inspired assurance-case schema and exporters.

This module reframes append-only evaluation evidence as a draft safety
argument.  It deliberately does not certify the top-level claim: evidence nodes
can be auto-populated from run outputs, while claims, assumptions, and defeaters
default to human review.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

ASSURANCE_SCHEMA_VERSION = "cc/assurance-case.v1"

NEEDS_HUMAN_REVIEW = "NEEDS HUMAN REVIEW"
AUTO_POPULATED_NOTE = (
    "Auto-populated from run data; adequacy, interpretation, and acceptance "
    "criteria require human review."
)

_MAX_EVIDENCE_ITEMS_PER_ROLE = 12
_MAX_FILE_BYTES = 5_000_000
_MAX_JSONL_RECORDS = 25


class ReviewStatus(str, Enum):
    """Review status for claims, assumptions, defeaters, and evidence."""

    NEEDS_HUMAN_REVIEW = NEEDS_HUMAN_REVIEW
    AUTO_POPULATED = "AUTO-POPULATED"
    HUMAN_REVIEWED = "HUMAN REVIEWED"
    REJECTED = "REJECTED"


class ClaimCategory(str, Enum):
    """Safety-argument claim categories used by the generated GSN tree."""

    SYSTEM_SAFETY_ARGUMENT = "system_safety_argument"
    COMPOSITION_RISK_BOUNDED = "composition_risk_bounded"
    DEPENDENCE_STRUCTURE_CHARACTERIZED = "dependence_structure_characterized"
    UNCERTAINTY_HONESTLY_QUANTIFIED = "uncertainty_honestly_quantified"


class EvidenceRole(str, Enum):
    """Roles for run-derived evidence within the assurance case."""

    FH_BOUNDS = "fh_bounds"
    CLIFF_CERTIFICATE = "cliff_certificate"
    CCF_CONSISTENCY_CHECK = "ccf_consistency_check"
    STATISTICAL_COVERAGE = "statistical_coverage"
    CLAIM_DECAY = "claim_decay"
    EXTREMAL_SCENARIO = "extremal_scenario"
    RUN_METADATA = "run_metadata"


class AssuranceModel(BaseModel):
    """Base config for assurance-case records."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class Context(AssuranceModel):
    """GSN context node: background that scopes a claim."""

    id: str
    description: str
    value: Any | None = None
    source_ref: str | None = None
    auto_populated: bool = False
    human_review_required: bool = False


class Assumption(AssuranceModel):
    """GSN assumption node: a premise that remains open until reviewed."""

    id: str
    statement: str
    rationale: str | None = None
    review_status: ReviewStatus = ReviewStatus.NEEDS_HUMAN_REVIEW
    human_review_required: bool = True


class Defeater(AssuranceModel):
    """Explicit challenge that could invalidate or weaken a claim."""

    id: str
    description: str
    mitigation_plan: str | None = None
    review_status: ReviewStatus = ReviewStatus.NEEDS_HUMAN_REVIEW
    human_review_required: bool = True
    resolved_by_evidence_ids: list[str] = Field(default_factory=list)


class Evidence(AssuranceModel):
    """GSN evidence/solution node harvested from run artifacts."""

    id: str
    role: EvidenceRole
    description: str
    source_ref: str
    data: Any
    review_status: ReviewStatus = ReviewStatus.AUTO_POPULATED
    auto_populated: bool = True
    source_is_actual_run_output: bool = True
    human_review_required: bool = True
    notes: str = AUTO_POPULATED_NOTE


class Strategy(AssuranceModel):
    """GSN strategy node: how a claim is decomposed or supported."""

    id: str
    description: str
    rationale: str
    supports_claim_id: str | None = None
    decomposes_into_claim_ids: list[str] = Field(default_factory=list)
    review_status: ReviewStatus = ReviewStatus.NEEDS_HUMAN_REVIEW
    human_review_required: bool = True


class ClaimBase(AssuranceModel):
    """Common policy for all claim nodes.

    Every claim must explicitly enumerate defeaters.  If a reviewer really
    believes no defeaters apply, they must say why via
    ``no_defeaters_justification``; omission is rejected.
    """

    id: str
    statement: str
    category: ClaimCategory
    strategy: Strategy | None = None
    contexts: list[Context] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    defeaters: list[Defeater] = Field(...)
    no_defeaters_justification: str | None = None
    review_status: ReviewStatus = ReviewStatus.NEEDS_HUMAN_REVIEW
    human_review_required: bool = True

    @model_validator(mode="after")
    def _require_explicit_defeater_position(self) -> ClaimBase:
        if not self.defeaters and not (
            self.no_defeaters_justification and self.no_defeaters_justification.strip()
        ):
            raise ValueError(
                "claim nodes must enumerate defeaters; an empty defeater list requires "
                "no_defeaters_justification"
            )
        return self


class SubClaim(ClaimBase):
    """GSN goal/sub-claim node."""


class TopClaim(ClaimBase):
    """Top-level assurance claim with a decomposed GSN strategy."""

    subclaims: list[SubClaim] = Field(default_factory=list)


class AssuranceCase(AssuranceModel):
    """Machine-readable assurance case rooted at one top claim."""

    schema_id: str = Field(default=ASSURANCE_SCHEMA_VERSION, alias="schema")
    id: str
    run_id: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_refs: list[str] = Field(default_factory=list)
    top_claim: TopClaim
    notes: list[str] = Field(default_factory=list)


def assurance_case_from_run(evidence_bundle: Mapping[str, Any] | str | Path) -> AssuranceCase:
    """Build a draft GSN-style assurance case from actual run outputs.

    Automated population is limited to evidence nodes extracted from the
    supplied bundle, referenced files, or run directory.  The top-level claim,
    sub-claim assertions, assumptions, and defeaters are intentionally marked
    ``NEEDS HUMAN REVIEW`` because the function cannot know the deployment
    context, risk tolerance, legal obligations, or whether the evidence is
    sufficient for a real safety decision.
    """

    payload, source_refs = _load_bundle_payload(evidence_bundle)
    run_id = _extract_run_id(payload)
    run_slug = _slug(run_id or "unknown-run")
    case_id = f"assurance-case-{run_slug}"

    # AUTOMATED: these evidence nodes are copied from concrete run outputs.
    # They are not transformed into accepted claims; reviewers must decide
    # whether the extracted data are complete, fresh, and relevant.
    fh_evidence = _evidence_from_matches(
        _collect_matches(payload, _FH_PATTERNS),
        role=EvidenceRole.FH_BOUNDS,
        id_prefix="ev-fh",
        description="Frechet-Hoeffding bounds or envelope outputs from the evaluation run.",
    )
    cliff_evidence = _evidence_from_matches(
        _collect_matches(payload, _CLIFF_PATTERNS),
        role=EvidenceRole.CLIFF_CERTIFICATE,
        id_prefix="ev-cliff",
        description="Cliff certificate or tail-dependence outputs from the evaluation run.",
    )
    ccf_evidence = _evidence_from_matches(
        _collect_matches(payload, _CCF_PATTERNS),
        role=EvidenceRole.CCF_CONSISTENCY_CHECK,
        id_prefix="ev-ccf",
        description="Common-cause-failure consistency or model-selection outputs from the run.",
    )
    coverage_evidence = _evidence_from_matches(
        _collect_matches(payload, _COVERAGE_PATTERNS),
        role=EvidenceRole.STATISTICAL_COVERAGE,
        id_prefix="ev-coverage",
        description="Coverage, confidence interval, or uncertainty quantification outputs.",
    )
    decay_evidence = _evidence_from_matches(
        _collect_matches(payload, _DECAY_PATTERNS),
        role=EvidenceRole.CLAIM_DECAY,
        id_prefix="ev-decay",
        description="Claim-decay policy, hazard covariates, or freshness-trigger artifacts.",
    )
    extremal_evidence = _evidence_from_matches(
        _collect_matches(payload, _EXTREMAL_PATTERNS),
        role=EvidenceRole.EXTREMAL_SCENARIO,
        id_prefix="ev-extremal",
        description="Extremal scenario atom-table or endpoint-distribution evidence.",
    )

    run_contexts = _run_contexts(payload, run_id)
    composition_claim_id = f"claim-{run_slug}-composition-risk-bounded"
    dependence_claim_id = f"claim-{run_slug}-dependence-characterized"
    uncertainty_claim_id = f"claim-{run_slug}-uncertainty-quantified"
    top_claim_id = f"claim-{run_slug}-top"

    composition_claim = SubClaim(
        id=composition_claim_id,
        category=ClaimCategory.COMPOSITION_RISK_BOUNDED,
        statement=(
            "Draft claim for human review: composition risk is bounded by the "
            "run's reported Frechet-Hoeffding outputs."
        ),
        strategy=Strategy(
            id=f"strategy-{run_slug}-composition",
            supports_claim_id=composition_claim_id,
            description="Use dependence-agnostic FH bounds as conservative composition evidence.",
            rationale=(
                "FH bounds can constrain feasible joint behavior from observed or estimated "
                "marginals, but they do not by themselves approve deployment risk."
            ),
        ),
        contexts=run_contexts,
        assumptions=[
            Assumption(
                id=f"assumption-{run_slug}-fh-inputs",
                statement=(
                    "The marginal rates, topology, and operating points used for FH bounds "
                    "match the evaluated system and intended deployment context."
                ),
            )
        ],
        evidence=[*fh_evidence, *extremal_evidence],
        defeaters=_composition_defeaters(run_slug, found_evidence=bool(fh_evidence)),
    )

    dependence_claim = SubClaim(
        id=dependence_claim_id,
        category=ClaimCategory.DEPENDENCE_STRUCTURE_CHARACTERIZED,
        statement=(
            "Draft claim for human review: dependence structure is characterized by "
            "available cliff certificates and CCF consistency checks."
        ),
        strategy=Strategy(
            id=f"strategy-{run_slug}-dependence",
            supports_claim_id=dependence_claim_id,
            description=(
                "Combine tail-dependence certificates with CCF/FH consistency checks as "
                "diagnostic evidence about dependence."
            ),
            rationale=(
                "Cliff and CCF outputs characterize observed or modeled dependence, but "
                "human reviewers must decide whether those diagnostics cover the threat model."
            ),
        ),
        contexts=run_contexts,
        assumptions=[
            Assumption(
                id=f"assumption-{run_slug}-dependence-stability",
                statement=(
                    "The dependence behavior measured during evaluation remains stable "
                    "under expected deployment traffic and adversarial adaptation."
                ),
            )
        ],
        evidence=[*cliff_evidence, *ccf_evidence],
        defeaters=_dependence_defeaters(
            run_slug,
            found_cliff=bool(cliff_evidence),
            found_ccf=bool(ccf_evidence),
        ),
    )

    uncertainty_claim = SubClaim(
        id=uncertainty_claim_id,
        category=ClaimCategory.UNCERTAINTY_HONESTLY_QUANTIFIED,
        statement=(
            "Draft claim for human review: uncertainty is honestly quantified by the "
            "run's reported statistical coverage and interval outputs."
        ),
        strategy=Strategy(
            id=f"strategy-{run_slug}-uncertainty",
            supports_claim_id=uncertainty_claim_id,
            description=(
                "Use coverage simulations, interval widths, and tolerance checks to expose "
                "sampling uncertainty rather than hiding it behind point estimates."
            ),
            rationale=(
                "Coverage and interval outputs are run-derived statistical evidence; they "
                "still require review for sample adequacy, assumptions, and acceptance thresholds."
            ),
        ),
        contexts=run_contexts,
        assumptions=[
            Assumption(
                id=f"assumption-{run_slug}-coverage-design",
                statement=(
                    "Coverage simulations, bootstrap settings, and confidence levels are "
                    "appropriate for the evaluation design and decision being made."
                ),
            )
        ],
        evidence=[*coverage_evidence, *decay_evidence],
        defeaters=_uncertainty_defeaters(run_slug, found_evidence=bool(coverage_evidence)),
    )

    subclaims = [composition_claim, dependence_claim, uncertainty_claim]

    # HUMAN REQUIRED: the generated top claim is only a review scaffold.
    # The function never auto-asserts that an AI system is safe, compliant, or
    # ready to deploy.
    top_claim = TopClaim(
        id=top_claim_id,
        category=ClaimCategory.SYSTEM_SAFETY_ARGUMENT,
        statement=(
            "Draft top-level claim for human review: the evaluated AI safety controls "
            "have an evidence-backed assurance argument for the stated run context."
        ),
        strategy=Strategy(
            id=f"strategy-{run_slug}-top",
            supports_claim_id=top_claim_id,
            decomposes_into_claim_ids=[claim.id for claim in subclaims],
            description=(
                "Decompose the top-level assurance argument into composition risk, "
                "dependence characterization, and uncertainty quantification sub-claims."
            ),
            rationale=(
                "This mirrors GSN-style safety-case practice: a high-level claim is "
                "supported by structured argument, context, assumptions, evidence, and "
                "explicit defeaters."
            ),
        ),
        contexts=run_contexts,
        assumptions=[
            Assumption(
                id=f"assumption-{run_slug}-human-approval",
                statement=(
                    "An accountable human reviewer will evaluate every claim, assumption, "
                    "evidence item, and defeater before relying on this assurance case."
                ),
            ),
            Assumption(
                id=f"assumption-{run_slug}-scope-match",
                statement=(
                    "The run scope, harm taxonomy, data sources, and operational controls "
                    "match the deployment decision under review."
                ),
            ),
        ],
        evidence=[],
        defeaters=[
            Defeater(
                id=f"defeater-{run_slug}-top-context",
                description=(
                    "Deployment context, intended use, risk tolerance, and accountable "
                    "approver may not be documented or may differ from the evaluation run."
                ),
                mitigation_plan="Human governance review must bind this case to a named deployment context.",
            ),
            Defeater(
                id=f"defeater-{run_slug}-top-staleness",
                description=(
                    "The evidence bundle may be incomplete, stale, selectively generated, "
                    "or disconnected from current model/guardrail versions."
                ),
                mitigation_plan="Check run manifests, version lineage, and transparency-log verification.",
            ),
            Defeater(
                id=f"defeater-{run_slug}-top-compliance",
                description=(
                    "A structured assurance argument is not a legal conclusion, audit "
                    "opinion, or certification of NIST AI RMF or ISO/IEC 42001 conformance."
                ),
                mitigation_plan="Route compliance conclusions to qualified legal/compliance reviewers.",
            ),
        ],
        subclaims=subclaims,
    )

    return AssuranceCase(
        id=case_id,
        run_id=run_id,
        source_refs=source_refs,
        top_claim=top_claim,
        notes=[
            "Evidence nodes were harvested from actual run outputs where matching fields existed.",
            "Claims, assumptions, and defeaters default to NEEDS HUMAN REVIEW.",
            "This draft does not certify safety, compliance, or conformance.",
        ],
    )


def export_assurance_case_jsonld(case: AssuranceCase) -> dict[str, Any]:
    """Export an assurance case as JSON-LD with an explicit graph."""

    graph: list[dict[str, Any]] = [
        {
            "@id": case.id,
            "@type": "AssuranceCase",
            "schema": case.schema_id,
            "runId": case.run_id,
            "createdAt": case.created_at,
            "sourceRefs": case.source_refs,
            "hasTopClaim": {"@id": case.top_claim.id},
            "notes": case.notes,
        }
    ]
    graph.extend(_claim_jsonld(case.top_claim))
    graph = _dedupe_graph_nodes(graph)

    return {
        "@context": {
            "cc": "https://github.com/Cubits11/cc-framework/schema#",
            "gsn": "https://scsc.uk/gsn#",
            "AssuranceCase": "cc:AssuranceCase",
            "TopClaim": "gsn:Goal",
            "SubClaim": "gsn:Goal",
            "Strategy": "gsn:Strategy",
            "Evidence": "gsn:Solution",
            "Assumption": "gsn:Assumption",
            "Context": "gsn:Context",
            "Defeater": "cc:Defeater",
            "hasTopClaim": {"@id": "cc:hasTopClaim", "@type": "@id"},
            "supportedBy": {"@id": "cc:supportedBy", "@type": "@id"},
            "hasStrategy": {"@id": "gsn:hasStrategy", "@type": "@id"},
            "hasEvidence": {"@id": "gsn:hasSolution", "@type": "@id"},
            "hasAssumption": {"@id": "gsn:hasAssumption", "@type": "@id"},
            "hasContext": {"@id": "gsn:hasContext", "@type": "@id"},
            "hasDefeater": {"@id": "cc:hasDefeater", "@type": "@id"},
        },
        "@id": case.id,
        "@type": "AssuranceCase",
        "@graph": graph,
    }


def export_assurance_case_markdown(case: AssuranceCase) -> str:
    """Export an assurance case as human-readable Markdown."""

    lines = [
        f"# Assurance Case: {case.id}",
        "",
        f"- **Schema:** `{case.schema_id}`",
        f"- **Run ID:** `{case.run_id or 'unknown'}`",
        f"- **Created:** `{case.created_at}`",
        "- **Status:** `NEEDS HUMAN REVIEW`",
        "",
        "> This draft structures run evidence as a GSN-inspired assurance argument. "
        "It does not certify safety, legal compliance, or standards conformance.",
        "",
    ]
    if case.source_refs:
        lines.append("## Source References")
        lines.extend(f"- `{source}`" for source in case.source_refs)
        lines.append("")

    lines.extend(_claim_markdown(case.top_claim, heading_level=2))
    return "\n".join(lines).rstrip() + "\n"


def write_assurance_case_exports(
    case: AssuranceCase,
    output_dir: str | Path,
    *,
    stem: str = "assurance_case",
) -> dict[str, str]:
    """Write JSON-LD and Markdown exports and return their paths."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonld_path = out / f"{stem}.jsonld"
    markdown_path = out / f"{stem}.md"
    jsonld_path.write_text(
        json.dumps(export_assurance_case_jsonld(case), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    markdown_path.write_text(export_assurance_case_markdown(case), encoding="utf-8")
    return {"jsonld": str(jsonld_path), "markdown": str(markdown_path)}


_FH_PATTERNS = (
    "fh",
    "frechet",
    "fréchet",
    "hoeffding",
    "envelope",
    "composition_bound",
    "lower_distribution",
    "upper_distribution",
    "endpoint_distribution",
)
_CLIFF_PATTERNS = (
    "cliff",
    "tail_dependence",
    "tail-dependence",
    "lambda_hat",
    "lambda_lower",
    "lambda_upper",
    "critical_value",
    "falsifier",
)
_CCF_PATTERNS = (
    "ccf",
    "common_cause",
    "common-cause",
    "within_fh",
    "fh_consistency",
    "beta_factor",
    "alpha_factor",
    "mgl",
    "model_recommendation",
)
_COVERAGE_PATTERNS = (
    "coverage",
    "confidence_interval",
    "nominal_coverage",
    "cluster_bootstrap_coverage",
    "within_tolerance",
    "bootstrap",
    "wilson",
    "newcombe",
    "bca",
    "uncertainty",
    "ci_lower",
    "ci_upper",
    "ci_table",
    "standard_error",
)
_DECAY_PATTERNS = (
    "claim_decay",
    "decay",
    "hazard",
    "half_life",
    "ttl",
    "freshness",
    "version_watch",
    "version_watch_set",
)
_EXTREMAL_PATTERNS = (
    "extremal_scenario",
    "extremal",
    "atom_table",
    "top_outcomes",
    "feasibility",
    "source_kernel",
    "excluded_evidence_fields",
)

_CONTEXT_KEYS = {
    "run_id",
    "seed",
    "seeds",
    "total_runs",
    "prompt_count",
    "episodes_per_config",
    "composition",
    "topology",
    "rails",
    "thresholds",
    "label",
    "n_samples",
    "monte_carlo_reps",
    "bootstrap_reps",
    "confidence_level",
    "alpha",
}


def _composition_defeaters(run_slug: str, *, found_evidence: bool) -> list[Defeater]:
    defeaters = [
        Defeater(
            id=f"defeater-{run_slug}-fh-scope",
            description=(
                "FH bounds constrain feasible joint rates, but do not by themselves "
                "establish acceptable residual risk, severity, or operational readiness."
            ),
            mitigation_plan="Reviewer must compare bounds with pre-approved risk tolerances.",
        ),
        Defeater(
            id=f"defeater-{run_slug}-fh-input-validity",
            description=(
                "FH conclusions can be invalid if marginals, topology, thresholds, or "
                "operating points were mis-specified or extracted from non-representative data."
            ),
            mitigation_plan="Reviewer must check data provenance and metric definitions.",
        ),
    ]
    if not found_evidence:
        defeaters.append(
            Defeater(
                id=f"defeater-{run_slug}-fh-missing",
                description="No FH/Frechet-Hoeffding bound outputs were found in the run bundle.",
                mitigation_plan="Generate or attach FH bounds before accepting this sub-claim.",
            )
        )
    return defeaters


def _dependence_defeaters(
    run_slug: str,
    *,
    found_cliff: bool,
    found_ccf: bool,
) -> list[Defeater]:
    defeaters = [
        Defeater(
            id=f"defeater-{run_slug}-dependence-shift",
            description=(
                "Observed dependence may shift under deployment traffic, adversarial "
                "adaptation, model updates, or guardrail configuration changes."
            ),
            mitigation_plan="Reviewer must confirm monitoring and re-evaluation triggers.",
        ),
        Defeater(
            id=f"defeater-{run_slug}-ccf-identification",
            description=(
                "CCF point estimates and consistency checks are assumption-laden diagnostics; "
                "passing an FH consistency check does not prove the true dependence model."
            ),
            mitigation_plan="Reviewer must document CCF assumptions and sensitivity analysis.",
        ),
    ]
    if not found_cliff:
        defeaters.append(
            Defeater(
                id=f"defeater-{run_slug}-cliff-missing",
                description="No cliff certificate or tail-dependence output was found.",
                mitigation_plan="Attach cliff certificates or explain why they are out of scope.",
            )
        )
    if not found_ccf:
        defeaters.append(
            Defeater(
                id=f"defeater-{run_slug}-ccf-missing",
                description="No CCF/FH consistency-check output was found.",
                mitigation_plan="Attach CCF consistency evidence or document why FH-only is used.",
            )
        )
    return defeaters


def _uncertainty_defeaters(run_slug: str, *, found_evidence: bool) -> list[Defeater]:
    defeaters = [
        Defeater(
            id=f"defeater-{run_slug}-coverage-transfer",
            description=(
                "Coverage results may not transfer to deployment if sampling, clustering, "
                "drift, or adversarial behavior differs from the simulation/evaluation design."
            ),
            mitigation_plan="Reviewer must compare statistical design assumptions to deployment.",
        ),
        Defeater(
            id=f"defeater-{run_slug}-uncertainty-threshold",
            description=(
                "Honest uncertainty quantification does not imply the interval width or "
                "coverage level is acceptable for the deployment decision."
            ),
            mitigation_plan="Reviewer must apply pre-defined acceptance criteria.",
        ),
    ]
    if not found_evidence:
        defeaters.append(
            Defeater(
                id=f"defeater-{run_slug}-coverage-missing",
                description="No statistical coverage or interval output was found in the run bundle.",
                mitigation_plan="Generate coverage/interval evidence before accepting this sub-claim.",
            )
        )
    return defeaters


def _load_bundle_payload(
    evidence_bundle: Mapping[str, Any] | str | Path,
) -> tuple[dict[str, Any], list[str]]:
    source_refs: list[str] = []
    if isinstance(evidence_bundle, (str, Path)):
        path = Path(evidence_bundle)
        source_refs.append(str(path))
        if path.is_dir():
            return {
                "run_directory": str(path),
                "loaded_files": _load_json_artifacts(path),
            }, source_refs
        return {"bundle_file": str(path), "payload": _read_artifact(path)}, source_refs

    payload = _jsonable(evidence_bundle)
    if not isinstance(payload, dict):
        payload = {"payload": payload}
    source_refs.append("provided_mapping")

    loaded_files: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "output_dir" and isinstance(value, str):
            out = Path(value)
            if out.is_dir():
                source_refs.append(str(out))
                loaded_files.update(_load_json_artifacts(out))
        elif key.endswith("_path") and isinstance(value, str):
            path = Path(value)
            if path.exists() and path.is_file():
                source_refs.append(str(path))
                loaded_files[str(path)] = _read_artifact(path)
    if loaded_files:
        payload = {**payload, "loaded_files": loaded_files}
    return payload, _dedupe(source_refs)


def _load_json_artifacts(directory: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if path.stat().st_size > _MAX_FILE_BYTES:
            continue
        rel = str(path.relative_to(directory))
        loaded[rel] = _read_artifact(path)
    return loaded


def _read_artifact(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _jsonable(json.loads(path.read_text(encoding="utf-8")))
    if suffix == ".jsonl":
        records: list[Any] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if line_number > _MAX_JSONL_RECORDS:
                    records.append({"truncated_after_records": _MAX_JSONL_RECORDS})
                    break
                if raw.strip():
                    records.append(_jsonable(json.loads(raw)))
        return records
    return path.read_text(encoding="utf-8")


def _extract_run_id(payload: Mapping[str, Any]) -> str | None:
    for path, value in _walk(payload):
        if path.endswith(".run_id") and isinstance(value, str) and value.strip():
            return value
    return None


def _run_contexts(payload: Mapping[str, Any], run_id: str | None) -> list[Context]:
    contexts = [
        Context(
            id="context-run-id",
            description="Run identifier extracted from the evidence bundle.",
            value=run_id or "unknown",
            source_ref="bundle.run_id",
            auto_populated=True,
        )
    ]
    for path, value in _walk(payload):
        leaf = path.rsplit(".", 1)[-1].lower()
        if leaf in {"composition", "prompt_count", "seed"}:
            contexts.append(
                Context(
                    id=f"context-{_slug(leaf)}",
                    description=f"Run metadata field `{leaf}` extracted from the bundle.",
                    value=_truncate_json(value),
                    source_ref=path,
                    auto_populated=True,
                )
            )
    deduped: dict[str, Context] = {}
    for context in contexts:
        deduped.setdefault(context.id, context)
    return list(deduped.values())


def _collect_matches(payload: Mapping[str, Any], patterns: Sequence[str]) -> list[tuple[str, Any]]:
    matches: list[tuple[str, Any]] = []
    seen_paths: set[str] = set()
    for path, value in _walk(payload):
        if len(matches) >= _MAX_EVIDENCE_ITEMS_PER_ROLE:
            break
        if not isinstance(value, Mapping):
            continue
        if not _mapping_matches(path, value, patterns):
            continue
        if path in seen_paths:
            continue
        snapshot = _compact_evidence_snapshot(value, patterns)
        if snapshot in ({}, []):
            continue
        matches.append((path, snapshot))
        seen_paths.add(path)
    return matches


def _mapping_matches(path: str, value: Mapping[str, Any], patterns: Sequence[str]) -> bool:
    path_l = path.lower()
    if _contains_any(path_l, patterns):
        return True
    role = value.get("role")
    if isinstance(role, str) and _contains_any(role.lower(), patterns):
        return True
    return any(_contains_any(str(key).lower(), patterns) for key in value)


def _compact_evidence_snapshot(value: Mapping[str, Any], patterns: Sequence[str]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for key, item in value.items():
        key_l = str(key).lower()
        if _contains_any(key_l, patterns) or key_l in _CONTEXT_KEYS:
            selected[str(key)] = _truncate_json(item)
    if selected:
        return selected
    return {str(key): _truncate_json(item) for key, item in list(value.items())[:8]}


def _evidence_from_matches(
    matches: Sequence[tuple[str, Any]],
    *,
    role: EvidenceRole,
    id_prefix: str,
    description: str,
) -> list[Evidence]:
    return [
        Evidence(
            id=f"{id_prefix}-{idx:03d}",
            role=role,
            description=description,
            source_ref=path,
            data=data,
        )
        for idx, (path, data) in enumerate(matches, start=1)
    ]


def _walk(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    items: list[tuple[str, Any]] = [(path, value)]
    if isinstance(value, Mapping):
        for key, child in value.items():
            items.extend(_walk(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            items.extend(_walk(child, f"{path}[{idx}]"))
    return items


def _contains_any(text: str, patterns: Sequence[str]) -> bool:
    return any(pattern in text for pattern in patterns)


def _truncate_json(value: Any, *, depth: int = 0) -> Any:
    value = _jsonable(value)
    if depth >= 3:
        if isinstance(value, (Mapping, list)):
            return "<truncated>"
        return value
    if isinstance(value, Mapping):
        items = list(value.items())
        truncated = {str(key): _truncate_json(item, depth=depth + 1) for key, item in items[:8]}
        if len(items) > 8:
            truncated["<truncated_keys>"] = len(items) - 8
        return truncated
    if isinstance(value, list):
        result = [_truncate_json(item, depth=depth + 1) for item in value[:8]]
        if len(value) > 8:
            result.append({"<truncated_items>": len(value) - 8})
        return result
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    try:
        json.dumps(value, allow_nan=False)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _claim_jsonld(claim: TopClaim | SubClaim) -> list[dict[str, Any]]:
    claim_type = "TopClaim" if isinstance(claim, TopClaim) else "SubClaim"
    graph = [
        {
            "@id": claim.id,
            "@type": claim_type,
            "statement": claim.statement,
            "category": claim.category.value,
            "reviewStatus": claim.review_status.value,
            "humanReviewRequired": claim.human_review_required,
            "hasStrategy": {"@id": claim.strategy.id} if claim.strategy else None,
            "hasEvidence": [{"@id": evidence.id} for evidence in claim.evidence],
            "hasAssumption": [{"@id": assumption.id} for assumption in claim.assumptions],
            "hasContext": [{"@id": context.id} for context in claim.contexts],
            "hasDefeater": [{"@id": defeater.id} for defeater in claim.defeaters],
        }
    ]
    if claim.strategy is not None:
        graph.append(_strategy_jsonld(claim.strategy))
    graph.extend(_context_jsonld(context) for context in claim.contexts)
    graph.extend(_assumption_jsonld(assumption) for assumption in claim.assumptions)
    graph.extend(_evidence_jsonld(evidence) for evidence in claim.evidence)
    graph.extend(_defeater_jsonld(defeater) for defeater in claim.defeaters)
    if isinstance(claim, TopClaim):
        graph[0]["supportedBy"] = [{"@id": subclaim.id} for subclaim in claim.subclaims]
        for subclaim in claim.subclaims:
            graph.extend(_claim_jsonld(subclaim))
    return graph


def _strategy_jsonld(strategy: Strategy) -> dict[str, Any]:
    return {
        "@id": strategy.id,
        "@type": "Strategy",
        "description": strategy.description,
        "rationale": strategy.rationale,
        "supportsClaim": {"@id": strategy.supports_claim_id}
        if strategy.supports_claim_id
        else None,
        "decomposesInto": [{"@id": claim_id} for claim_id in strategy.decomposes_into_claim_ids],
        "reviewStatus": strategy.review_status.value,
        "humanReviewRequired": strategy.human_review_required,
    }


def _context_jsonld(context: Context) -> dict[str, Any]:
    return {
        "@id": context.id,
        "@type": "Context",
        "description": context.description,
        "value": context.value,
        "sourceRef": context.source_ref,
        "autoPopulated": context.auto_populated,
        "humanReviewRequired": context.human_review_required,
    }


def _assumption_jsonld(assumption: Assumption) -> dict[str, Any]:
    return {
        "@id": assumption.id,
        "@type": "Assumption",
        "statement": assumption.statement,
        "rationale": assumption.rationale,
        "reviewStatus": assumption.review_status.value,
        "humanReviewRequired": assumption.human_review_required,
    }


def _evidence_jsonld(evidence: Evidence) -> dict[str, Any]:
    return {
        "@id": evidence.id,
        "@type": "Evidence",
        "role": evidence.role.value,
        "description": evidence.description,
        "sourceRef": evidence.source_ref,
        "data": evidence.data,
        "reviewStatus": evidence.review_status.value,
        "autoPopulated": evidence.auto_populated,
        "sourceIsActualRunOutput": evidence.source_is_actual_run_output,
        "humanReviewRequired": evidence.human_review_required,
        "notes": evidence.notes,
    }


def _defeater_jsonld(defeater: Defeater) -> dict[str, Any]:
    return {
        "@id": defeater.id,
        "@type": "Defeater",
        "description": defeater.description,
        "mitigationPlan": defeater.mitigation_plan,
        "reviewStatus": defeater.review_status.value,
        "humanReviewRequired": defeater.human_review_required,
        "resolvedByEvidenceIds": [
            {"@id": evidence_id} for evidence_id in defeater.resolved_by_evidence_ids
        ],
    }


def _claim_markdown(claim: TopClaim | SubClaim, *, heading_level: int) -> list[str]:
    heading = "#" * heading_level
    lines = [
        f"{heading} {claim.category.value}: `{claim.id}`",
        "",
        claim.statement,
        "",
        f"- **Review status:** `{claim.review_status.value}`",
        f"- **Human review required:** `{claim.human_review_required}`",
    ]
    if claim.strategy is not None:
        lines.extend(
            [
                "",
                f"{heading}# Strategy",
                "",
                f"**{claim.strategy.description}**",
                "",
                claim.strategy.rationale,
            ]
        )
    lines.extend(
        _node_list_markdown(
            "Contexts", [f"`{c.id}`: {c.description}" for c in claim.contexts], heading
        )
    )
    lines.extend(
        _node_list_markdown(
            "Assumptions",
            [f"`{a.id}`: {a.statement} (`{a.review_status.value}`)" for a in claim.assumptions],
            heading,
        )
    )
    lines.extend(
        _node_list_markdown(
            "Evidence",
            [
                f"`{e.id}` ({e.role.value}) from `{e.source_ref}`: {e.description}"
                for e in claim.evidence
            ],
            heading,
        )
    )
    lines.extend(
        _node_list_markdown(
            "Defeaters",
            [f"`{d.id}`: {d.description} (`{d.review_status.value}`)" for d in claim.defeaters],
            heading,
        )
    )
    if isinstance(claim, TopClaim) and claim.subclaims:
        lines.extend(["", f"{heading}# Subclaims", ""])
        for subclaim in claim.subclaims:
            lines.extend(_claim_markdown(subclaim, heading_level=heading_level + 2))
            lines.append("")
    return lines


def _node_list_markdown(title: str, items: Sequence[str], heading: str) -> list[str]:
    if not items:
        return ["", f"{heading}# {title}", "", "- None recorded."]
    return ["", f"{heading}# {title}", "", *[f"- {item}" for item in items]]


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "unknown"


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def _dedupe_graph_nodes(graph: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node in graph:
        node_id = node.get("@id")
        if isinstance(node_id, str):
            if node_id in seen:
                continue
            seen.add(node_id)
        deduped.append(node)
    return deduped


__all__ = [
    "ASSURANCE_SCHEMA_VERSION",
    "Assumption",
    "AssuranceCase",
    "ClaimCategory",
    "Context",
    "Defeater",
    "Evidence",
    "EvidenceRole",
    "ReviewStatus",
    "Strategy",
    "SubClaim",
    "TopClaim",
    "assurance_case_from_run",
    "export_assurance_case_jsonld",
    "export_assurance_case_markdown",
    "write_assurance_case_exports",
]
