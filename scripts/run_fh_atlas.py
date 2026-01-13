from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from experiments.fh_atlas.config import FHAtlasConfig
from experiments.fh_atlas.generate import run_fh_atlas


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FH dependence atlas experiments")
    parser.add_argument("--config", required=True, help="Path to FH atlas config JSON")
    args = parser.parse_args()

    config = FHAtlasConfig.from_json(Path(args.config))
    output_path = run_fh_atlas(config)
    print(f"FH atlas run complete: {output_path}")


if __name__ == "__main__":
    main()
