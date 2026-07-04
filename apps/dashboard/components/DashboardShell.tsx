"use client";

import { ClipboardList, FileCheck2, Network, ShieldAlert } from "lucide-react";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";
import { AssuranceCaseExplorer } from "./views/AssuranceCaseExplorer";
import { ClaimGovernanceView } from "./views/ClaimGovernanceView";
import { CompositionRiskView } from "./views/CompositionRiskView";
import { VerifyView } from "./views/VerifyView";
import type { ClaimObservatoryModel } from "../lib/claimObservatory";
import type { EnterpriseBundle } from "../lib/types";

type ViewKey = "composition" | "assurance" | "claim-governance" | "verify";

const views: Array<{ key: ViewKey; label: string; icon: ReactNode }> = [
  { key: "composition", label: "Composition Risk", icon: <ShieldAlert size={17} /> },
  { key: "assurance", label: "Assurance Case", icon: <Network size={17} /> },
  { key: "claim-governance", label: "Claim Governance", icon: <ClipboardList size={17} /> },
  { key: "verify", label: "Verify", icon: <FileCheck2 size={17} /> },
];

export function DashboardShell({
  initialBundle,
  initialClaimObservatoryModel,
}: {
  initialBundle?: EnterpriseBundle;
  initialClaimObservatoryModel?: ClaimObservatoryModel;
}) {
  const [activeView, setActiveView] = useState<ViewKey>("composition");
  const [bundle, setBundle] = useState<EnterpriseBundle | undefined>(initialBundle);
  const subtitle = useMemo(() => {
    if (!bundle) {
      return "No bundle loaded";
    }
    return `${bundle.bundle_id} · ${bundle.verification.tree_size} log records`;
  }, [bundle]);

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand">
          <h1>CC Evidence Dashboard</h1>
          <p>{subtitle}</p>
        </div>
        <nav className="tabs" aria-label="Evidence views">
          {views.map((view) => (
            <button
              aria-selected={activeView === view.key}
              className="tab"
              key={view.key}
              onClick={() => setActiveView(view.key)}
              role="tab"
              type="button"
            >
              {view.icon}
              <span>{view.label}</span>
            </button>
          ))}
        </nav>
      </header>

      <section className="content">
        {activeView === "composition" ? <CompositionRiskView bundle={bundle} /> : null}
        {activeView === "assurance" ? <AssuranceCaseExplorer bundle={bundle} /> : null}
        {activeView === "claim-governance" ? (
          <ClaimGovernanceView model={initialClaimObservatoryModel} />
        ) : null}
        {activeView === "verify" ? <VerifyView bundle={bundle} onBundleLoaded={setBundle} /> : null}
      </section>
    </main>
  );
}
