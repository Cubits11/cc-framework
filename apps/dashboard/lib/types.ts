export type CompositionRisk = {
  composition_rule: string;
  marginals: Array<{ name: string; probability: number }>;
  envelope: {
    event: string;
    lower: number;
    upper: number;
    width: number;
  };
  empirical: {
    estimate: number;
    label: string;
  };
  cliff_certificate: {
    regime: string;
    lambda_hat: number;
    ci: [number, number] | number[];
    critical_value: number;
    confidence_level: number;
    statement: string;
    falsifier: string;
  };
};

export type TreeClaim = {
  id: string;
  statement: string;
  category?: string;
  strategy?: TreeStrategy | null;
  contexts?: TreeContext[];
  assumptions?: TreeAssumption[];
  evidence?: TreeEvidence[];
  defeaters?: TreeDefeater[];
  subclaims?: TreeClaim[];
  review_status?: string;
};

export type TreeStrategy = {
  id: string;
  description: string;
  rationale?: string;
  decomposes_into_claim_ids?: string[];
  review_status?: string;
};

export type TreeContext = {
  id: string;
  description: string;
  value?: unknown;
};

export type TreeAssumption = {
  id: string;
  statement: string;
  review_status?: string;
};

export type TreeEvidence = {
  id: string;
  role?: string;
  description: string;
  source_ref?: string;
  review_status?: string;
};

export type TreeDefeater = {
  id: string;
  description: string;
  mitigation_plan?: string;
  review_status?: string;
};

export type AssuranceCase = {
  id: string;
  run_id?: string | null;
  top_claim: TreeClaim;
  notes?: string[];
};

export type MerkleProofStep = {
  side: "left" | "right";
  hash: string;
};

export type InclusionProof = {
  schema: string;
  hash_algorithm: string;
  record_id: number;
  tree_size: number;
  root_hash: string;
  leaf_hash: string;
  proof: MerkleProofStep[];
};

export type ConsistencyProof = {
  schema: string;
  hash_algorithm: string;
  old_size: number;
  new_size: number;
  old_root: string;
  new_root: string;
  proof: string[];
};

export type VerificationPayload = {
  hash_algorithm: string;
  trusted_root: string;
  tree_size: number;
  records: unknown[];
  canonical_records?: string[];
  leaf_hashes: string[];
  inclusion_proofs: InclusionProof[];
  consistency_proof: ConsistencyProof;
};

export type EnterpriseBundle = {
  schema: string;
  bundle_id: string;
  created_at?: string;
  composition_risk: CompositionRisk;
  assurance_case: AssuranceCase;
  verification: VerificationPayload;
  attestation?: Record<string, unknown>;
  enterprise_attestation?: Record<string, unknown>;
  metrics?: Record<string, unknown>;
  manifest?: Record<string, unknown>;
};

export type ClientVerificationResult = {
  ok: boolean;
  inclusionOk: boolean;
  consistencyOk: boolean;
  leafHashesOk: boolean;
  checkedRecords: number;
  rootHash: string;
  treeSize: number;
  errors: string[];
};
