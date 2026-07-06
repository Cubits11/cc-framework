package cc.promotion.tiering

import rego.v1

default allow := false

default computed_tier := "tier-4"

score_tolerance := 0.0005

allow if {
	input.request.action == "promote"
	count(deny) == 0
	input.request.target_tier == computed_tier
}

deny contains msg if {
	not input.evidence.non_claims
	msg := "promotion denied: non_claims array is mandatory"
}

deny contains msg if {
	count(input.evidence.non_claims) == 0
	msg := "promotion denied: non_claims array must not be empty"
}

deny contains msg if {
	overclaim_text(input.evidence.content.title)
	msg := "promotion denied: evidence title asserts an overclaim"
}

deny contains msg if {
	some fragment in object.get(input.evidence.content, "claim_fragments", [])
	overclaim_text(fragment.text)
	msg := sprintf("promotion denied: claim fragment %v asserts an overclaim", [fragment.fragment_id])
}

deny contains msg if {
	not confidence_score_matches
	msg := sprintf(
		"promotion denied: confidence_score %v does not equal round(0.24P + 0.24I + 0.18R + 0.16C + 0.10T + 0.08E, 3) = %v",
		[input.evidence.epistemic.confidence_score, expected_confidence],
	)
}

deny contains msg if {
	requested_rank := tier_rank[input.request.target_tier]
	computed_rank := tier_rank[computed_tier]
	requested_rank < computed_rank
	msg := sprintf(
		"promotion denied: requested tier %v is stronger than computed tier %v",
		[input.request.target_tier, computed_tier],
	)
}

deny contains msg if {
	input.request.target_tier == "tier-1"
	not tier1_component_thresholds
	msg := "promotion denied: tier-1 requires provenance_score >= 0.95, integrity_score == 1.0, and confidence_score >= 0.85"
}

deny contains msg if {
	input.request.target_tier == "tier-1"
	not tier1_review_separation
	msg := "promotion denied: tier-1 requires at least two approving human reviewers from distinct review pools, distinct from submitter and collector"
}

deny contains msg if {
	input.request.target_tier == "tier-1"
	not integrity_receipt_caveat
	msg := "promotion denied: tier-1 evidence must preserve the receipt-integrity-not-statistical-validity caveat"
}

deny contains msg if {
	exploratory_redteam_detected
	not confirmatory_protocol_binding_present
	msg := "promotion denied: exploratory red-team evidence requires pre-registered confirmatory protocol binding and model version hash increment"
}

deny contains msg if {
	some path in evidence_paths
	contains(path, "src/cc/claims/")
	msg := sprintf("promotion denied: claim lifecycle path is quarantined from evidence promotion: %v", [path])
}

expected_confidence := score if {
	c := input.evidence.epistemic.component_scores
	raw := (((((0.24 * c.provenance_score) + (0.24 * c.integrity_score)) + (0.18 * c.source_reliability_score)) + (0.16 * c.corroboration_score)) + (0.10 * c.temporal_validity_score)) + (0.08 * c.extraction_quality_score)
	score := round(raw * 1000) / 1000
}

confidence_score_matches if {
	diff := abs(input.evidence.epistemic.confidence_score - expected_confidence)
	diff <= score_tolerance
}

computed_tier := "tier-1" if {
	tier1_component_thresholds
}

computed_tier := "tier-2" if {
	not tier1_component_thresholds
	c := input.evidence.epistemic.component_scores
	c.provenance_score >= 0.80
	c.integrity_score >= 0.90
	input.evidence.epistemic.confidence_score >= 0.70
}

computed_tier := "tier-3" if {
	not tier1_component_thresholds
	not tier2_thresholds
	c := input.evidence.epistemic.component_scores
	c.provenance_score >= 0.55
	c.integrity_score >= 0.70
	input.evidence.epistemic.confidence_score >= 0.45
}

tier1_component_thresholds if {
	c := input.evidence.epistemic.component_scores
	c.provenance_score >= 0.95
	c.integrity_score == 1.0
	input.evidence.epistemic.confidence_score >= 0.85
}

tier2_thresholds if {
	c := input.evidence.epistemic.component_scores
	c.provenance_score >= 0.80
	c.integrity_score >= 0.90
	input.evidence.epistemic.confidence_score >= 0.70
}

tier1_review_separation if {
	approvals := [review |
		some review in input.reviews
		review.decision == "approve"
		review.actor_type == "person"
	]
	count(approvals) >= 2
	count({review.actor_id | some review in approvals}) >= 2
	count({review.review_pool | some review in approvals}) >= 2
	not submitter_or_collector_approved
	not policy_admin_exception_self_approved
}

submitter_or_collector_approved if {
	some review in input.reviews
	review.decision == "approve"
	review.actor_id == input.evidence.acquisition.submitted_by.actor_id
}

submitter_or_collector_approved if {
	some review in input.reviews
	review.decision == "approve"
	review.actor_id == input.evidence.provenance.collector.actor_id
}

policy_admin_exception_self_approved if {
	input.request.exception_id
	some review in input.reviews
	review.decision == "approve"
	review.actor_id == input.request.exception_requested_by
	review.role == "policy-admin"
}

integrity_receipt_caveat if {
	caveat := input.evidence.epistemic.receipt_integrity_caveat
	contains(lower(caveat), "integrity")
	contains(lower(caveat), "not")
	contains(lower(caveat), "statistical validity")
}

exploratory_redteam_detected if {
	input.evidence.evidence_class == "redteam_discovery"
}

exploratory_redteam_detected if {
	some path in evidence_paths
	contains(path, "src/cc/redteam/")
}

exploratory_redteam_detected if {
	input.evidence.extraction.method == "llm-assisted-extraction"
	object.get(input.evidence, "redteam_origin", false) == true
}

confirmatory_protocol_binding_present if {
	binding := input.evidence.confirmatory_binding
	binding.protocol_hash
	binding.model_version_hash
	binding.pre_registered_at
}

evidence_paths contains path if {
	refs := object.get(input.evidence.content, "storage_refs", {})
	path := object.get(refs, "raw_object_ref", "")
	path != ""
}

evidence_paths contains path if {
	refs := object.get(input.evidence.content, "storage_refs", {})
	path := object.get(refs, "normalized_text_ref", "")
	path != ""
}

evidence_paths contains path if {
	refs := object.get(input.evidence.content, "storage_refs", {})
	path := object.get(refs, "derivative_ref", "")
	path != ""
}

overclaim_text(text) if {
	value := lower(text)
	contains(value, "safe for deployment")
}

overclaim_text(text) if {
	value := lower(text)
	contains(value, "absolute safety")
}

overclaim_text(text) if {
	value := lower(text)
	contains(value, "guaranteed")
}

overclaim_text(text) if {
	value := lower(text)
	contains(value, "model truth")
}

overclaim_text(text) if {
	value := lower(text)
	contains(value, "causal inference without assumptions")
}

tier_rank := {
	"tier-1": 1,
	"tier-2": 2,
	"tier-3": 3,
	"tier-4": 4,
}
