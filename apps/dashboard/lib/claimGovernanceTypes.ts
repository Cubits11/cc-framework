export type GovernanceVerdict = "pass" | "needs_review" | "fail";
export type ClaimFreshnessStatus = "not_evaluated" | "fresh" | "degraded" | "expired";
export type SupportStrength = "weak" | "diagnostic" | "confirmatory" | "integrity_only";

export type CapsuleManifestFile = {
  filename: string;
  role: string;
  sha256: string;
  bytes: number;
};

export type CapsuleManifest = {
  schema_version: string;
  capsule_id?: string;
  report_id: string;
  governance_verdict?: GovernanceVerdict;
  pass_caveat?: string;
  report_receipt_sha256?: string;
  files: CapsuleManifestFile[];
};

export type CcReportClaim = {
  statement: string;
  allowed_claim_level: string;
  non_claims: string[];
};

export type CcReportMeasurement = {
  metric_family?: string;
  interval_method?: string;
  point_estimate?: number | null;
  confidence_level?: number | null;
  delta?: number | null;
  interval?: {
    lower: number;
    upper: number;
  };
  sample_sizes?: Record<string, number>;
};

export type CcReportEvidenceArtifact = {
  path: string;
  role: string;
  sha256: string;
  bytes: number;
};

export type CcReportReceipt = {
  canonical_hash?: string | null;
  canonicalization_method?: string;
  hash_algorithm?: string;
  previous_hash?: string | null;
};

export type CcReport = {
  schema_version: string;
  report_id: string;
  created_at?: string;
  claim: CcReportClaim;
  measurement?: CcReportMeasurement;
  evidence?: {
    artifacts?: CcReportEvidenceArtifact[];
    audit_log?: CcReportEvidenceArtifact | null;
    figure_manifest?: CcReportEvidenceArtifact | null;
  };
  receipt?: CcReportReceipt;
};

export type ClaimGovernanceReceiptAudit = {
  report_hash_verified: boolean | null;
  artifact_hashes_verified: boolean;
  canonical_hash: string | null;
  reason: string;
};

export type ClaimGovernanceEvidenceArtifactAudit = {
  path: string;
  role: string;
  sha256_expected: string;
  sha256_actual: string | null;
  bytes_expected: number | null;
  bytes_actual: number | null;
  status: string;
  reason: string;
};

export type ClaimGovernanceDecayAudit = {
  present: boolean;
  status: ClaimFreshnessStatus;
  reason: string;
  evaluated_at: string | null;
  trigger_summary: string[];
  non_claims: string[];
};

export type ClaimGovernanceScenarioAudit = {
  present: boolean;
  scenario_count: number;
  scenario_ids: string[];
  kinds: string[];
  infeasible_count: number;
  excluded_evidence_fields: Array<Record<string, unknown>>;
  non_claims: string[];
};

export type ClaimGovernanceBoundaryAudit = {
  claim_non_claim_count: number;
  artifact_non_claim_count: number;
  mandatory_non_claims_missing: string[];
  unresolved_defeaters_or_gaps: string[];
};

export type ClaimGovernanceConfirmatoryProtocolsAudit = {
  present: boolean;
  artifact_count: number;
  protocol_ids: string[];
  run_ids: string[];
  failed_count: number;
  review_count: number;
  failed_reasons: string[];
  review_reasons: string[];
  audits: Array<Record<string, unknown>>;
  non_claims: string[];
};

export type EnvelopeSupportSummary = {
  schema: string;
  support_edge_count: number;
  relation_counts: Record<string, number>;
  strength_counts: Partial<Record<SupportStrength, number>> & Record<string, number | undefined>;
  strongest_non_integrity_strength: SupportStrength | null;
  integrity_only_edges: number;
  unknown_role_refs: number;
  unsupported_role_refs: string[];
  review_edges: number;
};

export type ClaimGovernanceAudit = {
  schema: string;
  report_id: string;
  evaluated_at: string;
  verdict: GovernanceVerdict;
  allowed_claim_level: string;
  claim_statement: string;
  receipt: ClaimGovernanceReceiptAudit;
  evidence_artifacts: ClaimGovernanceEvidenceArtifactAudit[];
  decay: ClaimGovernanceDecayAudit;
  scenarios: ClaimGovernanceScenarioAudit;
  confirmatory_protocols: ClaimGovernanceConfirmatoryProtocolsAudit;
  boundary: ClaimGovernanceBoundaryAudit;
  required_human_review: boolean;
  reasons: string[];
  non_claims: string[];
  envelope_support: EnvelopeSupportSummary;
};

export type ClaimIdentity = {
  artifact_id: string;
  claim_id: string;
  subject_ref: string;
  source_report_id: string;
  source_report_schema: string;
  source_report_hash: string | null;
  created_at?: string | null;
  evaluated_at?: string | null;
};

export type ClaimFragment = {
  fragment_id: string;
  text: string;
  fragment_type: string;
};

export type ClaimProposition = {
  statement: string;
  allowed_claim_level: string;
  fragments: ClaimFragment[];
};

export type BoundaryEnvelope = {
  schema: string;
  scope?: Record<string, unknown>;
  assumptions?: string[];
  non_claims: string[];
  defeaters?: Array<Record<string, unknown>>;
  invalidation_conditions?: Array<Record<string, unknown>>;
  review_requirements?: Array<Record<string, unknown>>;
};

export type ArtifactRef = {
  artifact_id: string;
  subject_ref: string;
  role: string;
  path: string | null;
  sha256: string | null;
  bytes: number | null;
  schema?: string | null;
  status?: string | null;
  reason?: string | null;
  created_at?: string | null;
  evaluated_at?: string | null;
  metadata?: Record<string, unknown>;
};

export type SupportEdge = {
  source_artifact_id: string;
  target_claim_fragment: string;
  relation: string;
  strength: SupportStrength;
  non_claims: string[];
  rationale: string | null;
};

export type SupportGraph = {
  schema: string;
  evidence_refs: ArtifactRef[];
  scenario_refs: ArtifactRef[];
  decay_refs: ArtifactRef[];
  receipt_refs: ArtifactRef[];
  review_refs: ArtifactRef[];
  support_edges: SupportEdge[];
};

export type GovernanceState = {
  schema: string;
  verdict: GovernanceVerdict | "not_evaluated";
  verifier_schema: string | null;
  evaluated_at: string | null;
  freshness_status: ClaimFreshnessStatus | null;
  required_human_review: boolean;
  reasons: string[];
  support_summary: EnvelopeSupportSummary;
};

export type ClaimEnvelope = {
  schema: string;
  identity: ClaimIdentity;
  proposition: ClaimProposition;
  boundary: BoundaryEnvelope;
  support_graph: SupportGraph;
  governance_state: GovernanceState;
};

export function isGovernanceVerdict(value: unknown): value is GovernanceVerdict {
  return value === "pass" || value === "needs_review" || value === "fail";
}
