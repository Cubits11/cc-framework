import type {
  CapsuleManifest,
  CapsuleManifestFile,
  CcReport,
  ClaimEnvelope,
  ClaimFreshnessStatus,
  ClaimGovernanceAudit,
  ClaimGovernanceConfirmatoryProtocolsAudit,
  ClaimGovernanceDecayAudit,
  ClaimGovernanceEvidenceArtifactAudit,
  ClaimGovernanceReceiptAudit,
  ClaimGovernanceScenarioAudit,
  EnvelopeSupportSummary,
  GovernanceVerdict,
  SupportEdge,
} from "./claimGovernanceTypes";
import { isGovernanceVerdict } from "./claimGovernanceTypes";

export const DEFAULT_PASS_CAVEAT =
  "PASS means internal consistency under verifier rules; it does not mean the AI system is safe in deployment.";

export type ClaimObservatoryModel = {
  claimId: string;
  reportId: string;
  sourceReportHash: string | null;
  claimStatement: string;
  allowedClaimLevel: string;
  governanceVerdict: GovernanceVerdict;
  governanceVerdictLabel: string;
  verifierSchema: string;
  evaluatedAt: string | null;
  freshnessStatus: ClaimFreshnessStatus;
  requiredHumanReview: boolean;
  passCaveat: string;
  reasons: string[];
  nonClaims: string[];
  supportSummary: EnvelopeSupportSummary;
  supportEdges: SupportEdge[];
  evidenceArtifacts: ClaimGovernanceEvidenceArtifactAudit[];
  manifestFiles: CapsuleManifestFile[];
  receipt: ClaimGovernanceReceiptAudit;
  decay: ClaimGovernanceDecayAudit;
  scenarios: ClaimGovernanceScenarioAudit;
  confirmatoryProtocols: ClaimGovernanceConfirmatoryProtocolsAudit;
};

export function buildClaimObservatoryModel(input: {
  report: CcReport;
  audit: ClaimGovernanceAudit;
  envelope: ClaimEnvelope;
  manifest: CapsuleManifest;
}): ClaimObservatoryModel {
  const { report, audit, envelope, manifest } = input;
  enforceContinuity({ report, audit, envelope, manifest });

  if (!isGovernanceVerdict(audit.verdict)) {
    continuityError("audit verdict is not supported by the claim observatory bridge");
  }
  if (!envelope.governance_state.verifier_schema) {
    continuityError("envelope governance_state verifier schema is missing");
  }
  if (!envelope.governance_state.freshness_status) {
    continuityError("envelope governance_state freshness status is missing");
  }

  return {
    claimId: envelope.identity.claim_id,
    reportId: report.report_id,
    sourceReportHash: envelope.identity.source_report_hash ?? audit.receipt.canonical_hash ?? null,
    claimStatement: envelope.proposition.statement,
    allowedClaimLevel: envelope.proposition.allowed_claim_level,
    governanceVerdict: audit.verdict,
    governanceVerdictLabel: governanceVerdictLabel(audit.verdict),
    verifierSchema: envelope.governance_state.verifier_schema,
    evaluatedAt: envelope.governance_state.evaluated_at ?? audit.evaluated_at,
    freshnessStatus: envelope.governance_state.freshness_status,
    requiredHumanReview: envelope.governance_state.required_human_review,
    passCaveat: manifest.pass_caveat ?? DEFAULT_PASS_CAVEAT,
    reasons: [...audit.reasons],
    nonClaims: dedupeStrings([
      ...audit.non_claims,
      ...envelope.boundary.non_claims,
      ...report.claim.non_claims,
    ]),
    supportSummary: envelope.governance_state.support_summary,
    supportEdges: [...envelope.support_graph.support_edges],
    evidenceArtifacts: [...audit.evidence_artifacts],
    manifestFiles: [...manifest.files],
    receipt: audit.receipt,
    decay: audit.decay,
    scenarios: audit.scenarios,
    confirmatoryProtocols: audit.confirmatory_protocols,
  };
}

function enforceContinuity({
  report,
  audit,
  envelope,
  manifest,
}: {
  report: CcReport;
  audit: ClaimGovernanceAudit;
  envelope: ClaimEnvelope;
  manifest: CapsuleManifest;
}) {
  if (audit.schema !== envelope.governance_state.verifier_schema) {
    continuityError("audit verifier schema does not match envelope governance_state verifier schema");
  }
  if (audit.verdict !== envelope.governance_state.verdict) {
    continuityError("audit verdict does not match envelope governance_state verdict");
  }
  if (audit.required_human_review !== envelope.governance_state.required_human_review) {
    continuityError(
      "audit required_human_review does not match envelope governance_state required_human_review",
    );
  }
  if (audit.decay.status !== envelope.governance_state.freshness_status) {
    continuityError("audit decay status does not match envelope governance_state freshness_status");
  }
  if (
    JSON.stringify(audit.envelope_support) !==
    JSON.stringify(envelope.governance_state.support_summary)
  ) {
    continuityError("audit envelope_support does not match envelope governance_state support_summary");
  }
  if (report.report_id !== audit.report_id) {
    continuityError("report id does not match audit report id");
  }
  if (report.report_id !== envelope.identity.source_report_id) {
    continuityError("report id does not match envelope identity source_report_id");
  }
  if (manifest.report_id !== report.report_id) {
    continuityError("manifest report id does not match report id");
  }
}

function governanceVerdictLabel(verdict: GovernanceVerdict): string {
  if (verdict === "pass") {
    return "PASS under verifier rules";
  }
  if (verdict === "needs_review") {
    return "Needs review";
  }
  return "Failed governance";
}

function continuityError(message: string): never {
  throw new Error(`Claim observatory continuity error: ${message}.`);
}

function dedupeStrings(values: string[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const value of values) {
    const item = value.trim();
    if (!item || seen.has(item)) {
      continue;
    }
    seen.add(item);
    out.push(item);
  }
  return out;
}
