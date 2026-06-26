import { AlertTriangle, ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import type {
  EnterpriseBundle,
  TreeAssumption,
  TreeClaim,
  TreeContext,
  TreeDefeater,
  TreeEvidence,
  TreeStrategy,
} from "../../lib/types";

type GraphNode = {
  id: string;
  kind: "claim" | "strategy" | "context" | "assumption" | "evidence" | "defeater";
  title: string;
  detail?: string;
  children?: GraphNode[];
};

export function AssuranceCaseExplorer({ bundle }: { bundle?: EnterpriseBundle }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());

  if (!bundle) {
    return (
      <div className="panel">
        <div className="section-head">
          <div>
            <h2>Assurance Case</h2>
            <p>Load an enterprise evidence bundle in the Verify view.</p>
          </div>
        </div>
        <div className="empty">Awaiting bundle data.</div>
      </div>
    );
  }

  const root = claimToNode(bundle.assurance_case.top_claim);
  const toggle = (id: string) => {
    setCollapsed((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <div className="panel">
      <div className="section-head">
        <div>
          <h2>Assurance Case</h2>
          <p>GSN-style claims are expanded into strategy, context, assumptions, evidence, and defeaters.</p>
        </div>
        <span className="run-chip">{bundle.assurance_case.id}</span>
      </div>
      <div className="tree-panel">
        <div className="tree" role="tree">
          <TreeNode collapsed={collapsed} node={root} onToggle={toggle} />
        </div>
      </div>
    </div>
  );
}

function TreeNode({
  collapsed,
  node,
  onToggle,
}: {
  collapsed: Set<string>;
  node: GraphNode;
  onToggle: (id: string) => void;
}) {
  const hasChildren = Boolean(node.children?.length);
  const isCollapsed = collapsed.has(node.id);
  const className = `tree-node ${node.kind}`;

  return (
    <div className={className} role="treeitem" aria-expanded={hasChildren ? !isCollapsed : undefined}>
      <div className="node-main">
        {hasChildren ? (
          <button
            aria-label={isCollapsed ? `Expand ${node.id}` : `Collapse ${node.id}`}
            className="icon-button"
            onClick={() => onToggle(node.id)}
            type="button"
          >
            {isCollapsed ? <ChevronRight size={17} /> : <ChevronDown size={17} />}
          </button>
        ) : (
          <span />
        )}
        <div>
          <div className="node-kicker">
            {node.kind === "defeater" ? <AlertTriangle size={14} aria-hidden="true" /> : null} {node.kind}
          </div>
          <p className="node-title">{node.title}</p>
          {node.detail ? <p className="hash">{node.detail}</p> : null}
        </div>
      </div>
      {hasChildren && !isCollapsed ? (
        <div className="children" role="group">
          {node.children?.map((child) => (
            <TreeNode collapsed={collapsed} key={child.id} node={child} onToggle={onToggle} />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function claimToNode(claim: TreeClaim): GraphNode {
  const children: GraphNode[] = [];
  if (claim.strategy) {
    children.push(strategyToNode(claim.strategy));
  }
  children.push(...(claim.contexts ?? []).map(contextToNode));
  children.push(...(claim.assumptions ?? []).map(assumptionToNode));
  children.push(...(claim.evidence ?? []).map(evidenceToNode));
  children.push(...(claim.defeaters ?? []).map(defeaterToNode));
  children.push(...(claim.subclaims ?? []).map(claimToNode));

  return {
    id: claim.id,
    kind: "claim",
    title: claim.statement,
    detail: claim.review_status ?? claim.category,
    children,
  };
}

function strategyToNode(strategy: TreeStrategy): GraphNode {
  return {
    id: strategy.id,
    kind: "strategy",
    title: strategy.description,
    detail: strategy.rationale,
  };
}

function contextToNode(context: TreeContext): GraphNode {
  return {
    id: context.id,
    kind: "context",
    title: context.description,
    detail: context.value === undefined ? undefined : JSON.stringify(context.value),
  };
}

function assumptionToNode(assumption: TreeAssumption): GraphNode {
  return {
    id: assumption.id,
    kind: "assumption",
    title: assumption.statement,
    detail: assumption.review_status,
  };
}

function evidenceToNode(evidence: TreeEvidence): GraphNode {
  return {
    id: evidence.id,
    kind: "evidence",
    title: evidence.description,
    detail: evidence.source_ref ?? evidence.role,
  };
}

function defeaterToNode(defeater: TreeDefeater): GraphNode {
  return {
    id: defeater.id,
    kind: "defeater",
    title: defeater.description,
    detail: defeater.mitigation_plan,
  };
}
