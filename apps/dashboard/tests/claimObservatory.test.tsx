import { render, screen } from "@testing-library/react";
import { describe, expect, test } from "vitest";
import { ClaimGovernanceView } from "../components/views/ClaimGovernanceView";
import { buildClaimObservatoryModel } from "../lib/claimObservatory";
import type {
  CapsuleManifest,
  CcReport,
  ClaimEnvelope,
  ClaimGovernanceAudit,
} from "../lib/claimGovernanceTypes";
import auditFixture from "../../../examples/claim_governance_capsule/expected/claim_governance_audit.json";
import envelopeFixture from "../../../examples/claim_governance_capsule/expected/claim_envelope.json";
import reportFixture from "../../../examples/claim_governance_capsule/expected/cc_report.json";
import manifestFixture from "../../../examples/claim_governance_capsule/manifest.expected.json";

const report = reportFixture as CcReport;
const audit = auditFixture as ClaimGovernanceAudit;
const envelope = envelopeFixture as ClaimEnvelope;
const manifest = manifestFixture as CapsuleManifest;

function buildFixtureModel() {
  return buildClaimObservatoryModel({ report, audit, envelope, manifest });
}

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

describe("claim observatory adapter", () => {
  test("buildClaimObservatoryModel loads deterministic capsule artifacts", () => {
    const model = buildFixtureModel();

    expect(model.claimStatement).toMatch(/\S/);
    expect(model.allowedClaimLevel).toBe("bounded_empirical");
    expect(model.governanceVerdict).toBe("pass");
    expect(model.governanceVerdictLabel).toBe("PASS under verifier rules");
    expect(model.freshnessStatus).toBe("fresh");
    expect(model.requiredHumanReview).toBe(false);
    expect(model.verifierSchema).toBe("cc/claim-governance-audit.v1");
    expect(model.nonClaims.length).toBeGreaterThan(0);
    expect(model.supportEdges.length).toBeGreaterThan(0);
    expect(model.manifestFiles.length).toBeGreaterThan(0);
  });

  test("adapter preserves semantic continuity", () => {
    const model = buildFixtureModel();

    expect(model.verifierSchema).toBe(audit.schema);
    expect(model.governanceVerdict).toBe(envelope.governance_state.verdict);
    expect(model.requiredHumanReview).toBe(
      envelope.governance_state.required_human_review,
    );
    expect(model.freshnessStatus).toBe(audit.decay.status);
  });

  test("adapter rejects broken verifier schema continuity", () => {
    const brokenEnvelope = clone(envelope);
    (brokenEnvelope.governance_state as { verifier_schema: string | null }).verifier_schema =
      null;

    expect(() =>
      buildClaimObservatoryModel({ report, audit, envelope: brokenEnvelope, manifest }),
    ).toThrow(/verifier schema/i);
  });

  test("adapter rejects verdict mismatch", () => {
    const brokenAudit = clone(audit);
    brokenAudit.verdict = "fail";

    expect(() =>
      buildClaimObservatoryModel({ report, audit: brokenAudit, envelope, manifest }),
    ).toThrow(/verdict/i);
  });

  test("adapter avoids banned safety language in computed labels", () => {
    const model = buildFixtureModel();
    const labels = [
      model.governanceVerdictLabel,
      model.freshnessStatus,
      model.requiredHumanReview ? "human review required" : "human review not required by verifier",
    ]
      .join(" ")
      .toLowerCase();

    for (const phrase of [
      "certified safe",
      "approved for deployment",
      "deployment safe",
      "verified safe",
    ]) {
      expect(labels).not.toContain(phrase);
    }
  });
});

describe("ClaimGovernanceView", () => {
  test("renders the read-only claim governance bridge without dangerous approval labels", () => {
    const model = buildFixtureModel();

    render(<ClaimGovernanceView model={model} />);

    expect(screen.getByText(model.claimStatement)).toBeInTheDocument();
    expect(screen.getByText("PASS under verifier rules")).toBeInTheDocument();
    expect(screen.getByText("This claim does NOT say")).toBeInTheDocument();
    expect(screen.getByText("cc/claim-governance-audit.v1")).toBeInTheDocument();
    expect(screen.getAllByText("fresh").length).toBeGreaterThan(0);
    expect(screen.getByText("bounded_empirical")).toBeInTheDocument();
    expect(screen.getAllByText(/deployment[- ]safety/i).length).toBeGreaterThan(0);

    const renderedText = document.body.textContent?.toLowerCase() ?? "";
    for (const phrase of ["certified safe", "approved for deployment", "verified safe"]) {
      expect(renderedText).not.toContain(phrase);
    }
  });
});
