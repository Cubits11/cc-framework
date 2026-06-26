import type { EnterpriseBundle } from "../../lib/types";

export function CompositionRiskView({ bundle }: { bundle?: EnterpriseBundle }) {
  if (!bundle) {
    return <Empty title="Composition Risk" />;
  }

  const risk = bundle.composition_risk;
  const lower = risk.envelope.lower;
  const upper = risk.envelope.upper;
  const estimate = risk.empirical.estimate;
  const certificate = risk.cliff_certificate;
  const lambda = certificate.lambda_hat;
  const ciLow = certificate.ci[0];
  const ciHigh = certificate.ci[1];

  const width = 760;
  const height = 420;
  const margin = { top: 28, right: 34, bottom: 54, left: 62 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const x = (value: number) => margin.left + clamp01(value) * plotWidth;
  const y = (value: number) => margin.top + (1 - clamp01(value)) * plotHeight;
  const envelopeY = y(upper);
  const envelopeHeight = Math.max(4, y(lower) - y(upper));

  return (
    <div className="panel">
      <div className="section-head">
        <div>
          <h2>Composition Risk</h2>
          <p>
            Fréchet-Hoeffding bounds define the feasible composed rate under the stated
            marginals; the plotted point is the empirical run estimate.
          </p>
        </div>
        <span className="run-chip">{bundle.bundle_id}</span>
      </div>

      <div className="risk-grid">
        <div className="chart-wrap">
          <svg
            aria-label="Frechet-Hoeffding envelope with empirical estimate"
            className="risk-chart"
            role="img"
            viewBox={`0 0 ${width} ${height}`}
          >
            <rect
              fill="#ffffff"
              height={plotHeight}
              stroke="#cad7d0"
              width={plotWidth}
              x={margin.left}
              y={margin.top}
            />
            {[0, 0.25, 0.5, 0.75, 1].map((tick) => (
              <g key={tick}>
                <line
                  stroke="#e2e8e4"
                  x1={margin.left}
                  x2={margin.left + plotWidth}
                  y1={y(tick)}
                  y2={y(tick)}
                />
                <text className="axis-label" textAnchor="end" x={margin.left - 10} y={y(tick) + 4}>
                  {tick.toFixed(2)}
                </text>
              </g>
            ))}
            {[0, 0.2, 0.4, 0.6, 0.8, 1].map((tick) => (
              <g key={tick}>
                <line
                  stroke="#edf2ef"
                  x1={x(tick)}
                  x2={x(tick)}
                  y1={margin.top}
                  y2={margin.top + plotHeight}
                />
                <text className="axis-label" textAnchor="middle" x={x(tick)} y={height - 22}>
                  {tick.toFixed(1)}
                </text>
              </g>
            ))}
            <rect
              fill="#dcefeb"
              height={envelopeHeight}
              opacity="0.9"
              stroke="#1f7a76"
              strokeWidth="2"
              width={plotWidth}
              x={margin.left}
              y={envelopeY}
            />
            <rect
              fill="#f1c98c"
              height={plotHeight}
              opacity="0.42"
              width={Math.max(3, x(ciHigh) - x(ciLow))}
              x={x(ciLow)}
              y={margin.top}
            />
            <line
              stroke="#c47a22"
              strokeDasharray="6 6"
              strokeWidth="3"
              x1={x(certificate.critical_value)}
              x2={x(certificate.critical_value)}
              y1={margin.top}
              y2={margin.top + plotHeight}
            />
            <circle cx={x(lambda)} cy={y(estimate)} fill="#405f9f" r="8" stroke="#ffffff" strokeWidth="3" />
            <text className="axis-label" textAnchor="middle" x={margin.left + plotWidth / 2} y={height - 4}>
              tail-dependence λ
            </text>
            <text
              className="axis-label"
              textAnchor="middle"
              transform={`translate(18 ${margin.top + plotHeight / 2}) rotate(-90)`}
            >
              composed event probability
            </text>
            <text fill="#1f7a76" fontSize="14" x={margin.left + 14} y={envelopeY + 24}>
              FH feasible envelope
            </text>
            <text fill="#405f9f" fontSize="14" x={x(lambda) + 12} y={y(estimate) - 10}>
              empirical estimate
            </text>
          </svg>
        </div>

        <aside className="side-panel">
          <div className="metric-list">
            <div className="metric">
              <span>Composition rule</span>
              <strong>{risk.composition_rule}</strong>
            </div>
            <div className="metric">
              <span>FH envelope</span>
              <strong>
                {formatPct(lower)} to {formatPct(upper)}
              </strong>
            </div>
            <div className="metric">
              <span>Empirical estimate</span>
              <strong>{formatPct(estimate)}</strong>
            </div>
            <div className="metric">
              <span>Cliff certificate</span>
              <strong>{certificate.regime}</strong>
            </div>
            {risk.marginals.map((marginal) => (
              <div className="metric" key={marginal.name}>
                <span>{marginal.name}</span>
                <strong>{formatPct(marginal.probability)}</strong>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}

function Empty({ title }: { title: string }) {
  return (
    <div className="panel">
      <div className="section-head">
        <div>
          <h2>{title}</h2>
          <p>Load an enterprise evidence bundle in the Verify view.</p>
        </div>
      </div>
      <div className="empty">Awaiting bundle data.</div>
    </div>
  );
}

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value));
}

function formatPct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}
