"""
Real-time metrics dashboard for experiments
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ExperimentDashboard:
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.metrics_file = self.results_dir / "metrics.json"

    def update(self, metrics: dict[str, Any]):
        """Update dashboard with latest metrics"""
        metrics["updated_at"] = datetime.now().isoformat()

        # Load existing metrics
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                history = json.load(f)
        else:
            history = {"runs": []}

        history["runs"].append(metrics)
        history["latest"] = metrics

        with open(self.metrics_file, "w") as f:
            json.dump(history, f, indent=2)

        # Print summary
        print(f"[{metrics['updated_at']}]")
        print(f"  Sessions: {metrics.get('n_sessions', 0)}")
        print(f"  J-stat: {metrics.get('j_statistic', 0):.4f}")
        print(f"  CC_max: {metrics.get('cc_max', 0):.4f}")
        print(f"  CI width: {metrics.get('ci_width', 0):.4f}")
