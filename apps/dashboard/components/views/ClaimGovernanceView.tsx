import type { ClaimObservatoryModel } from "../../lib/claimObservatory";

export function ClaimGovernanceView({ model }: { model?: ClaimObservatoryModel }) {
  if (!model) {
    return (
      <div className="panel">
        <div className="section-head">
          <div>
            <h2>Claim Governance</h2>
            <p>Load a claim-governance capsule model to inspect verifier-scoped claim semantics.</p>
          </div>
        </div>
        <div className="empty">Awaiting claim-governance capsule data.</div>
      </div>
    );
  }

  const nonClaimPreview = model.nonClaims.slice(0, 5);
  const receiptHash = model.receipt.canonical_hash ?? model.sourceReportHash;
  const relationCounts = Object.entries(model.supportSummary.relation_counts);
  const unsupportedRoleCount = model.supportSummary.unsupported_role_refs.length;

  return (
    <div className="panel">
      <div className="section-head">
        <div>
          <h2>Claim Governance</h2>
          <p>{model.claimStatement}</p>
        </div>
        <span className="run-chip">{model.claimId}</span>
      </div>

      <div className="governance-banner">
        <strong>{model.governanceVerdictLabel}</strong>
        <span>{model.passCaveat}</span>
        <span>not a deployment-safety proof</span>
      </div>

      <div className="governance-grid">
        <section className="governance-card wide">
          <div className="node-kicker">Claim header</div>
          <div className="detail-grid">
            <Detail label="Allowed claim level" value={model.allowedClaimLevel} />
            <Detail label="Freshness status" value={model.freshnessStatus} />
            <Detail
              label="Verifier-required human review"
              value={model.requiredHumanReview ? "yes" : "no"}
            />
            <Detail label="Verifier schema" value={model.verifierSchema} />
            <Detail label="Evaluated at" value={model.evaluatedAt ?? "missing"} />
            <Detail label="Report id" value={model.reportId} />
          </div>
        </section>

        <section className="governance-card">
          <div className="node-kicker">Support summary</div>
          <div className="metric-list compact">
            <Metric label="Support edges" value={String(model.supportSummary.support_edge_count)} />
            <Metric
              label="Strongest non-integrity strength"
              value={model.supportSummary.strongest_non_integrity_strength ?? "none"}
            />
            <Metric label="Integrity-only edges" value={String(model.supportSummary.integrity_only_edges)} />
            <Metric label="Unknown role refs" value={String(model.supportSummary.unknown_role_refs)} />
            <Metric label="Unsupported role refs" value={String(unsupportedRoleCount)} />
          </div>
        </section>

        <section className="governance-card">
          <div className="node-kicker">Relation counts</div>
          {relationCounts.length ? (
            <ul className="plain-list">
              {relationCounts.map(([relation, count]) => (
                <li key={relation}>
                  <span>{relation}</span>
                  <strong>{count}</strong>
                </li>
              ))}
            </ul>
          ) : (
            <p className="hash">No support relations were reported.</p>
          )}
        </section>

        <section className="governance-card wide">
          <div className="node-kicker">This claim does NOT say</div>
          <ul className="non-claim-list">
            {nonClaimPreview.map((nonClaim) => (
              <li key={nonClaim}>{nonClaim}</li>
            ))}
          </ul>
          <p className="hash">{model.nonClaims.length} total non-claims preserved by the bridge.</p>
        </section>

        <section className="governance-card">
          <div className="node-kicker">Evidence summary</div>
          <div className="metric-list compact">
            <Metric label="Artifact audits" value={String(model.evidenceArtifacts.length)} />
            <Metric label="Manifest files" value={String(model.manifestFiles.length)} />
            <Metric label="Receipt hash" value={receiptHash ?? "missing"} />
          </div>
        </section>

        <section className="governance-card">
          <div className="node-kicker">Decay summary</div>
          <div className="metric-list compact">
            <Metric label="Freshness status" value={model.decay.status} />
            <Metric
              label="Evaluated at"
              value={model.decay.evaluated_at ?? model.evaluatedAt ?? "missing"}
            />
          </div>
          {model.decay.trigger_summary.length ? (
            <ul className="plain-list">
              {model.decay.trigger_summary.map((trigger) => (
                <li key={trigger}>
                  <span>{trigger}</span>
                </li>
              ))}
            </ul>
          ) : null}
        </section>
      </div>
    </div>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
