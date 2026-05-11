from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_imetad_dataset import main as imetad_main
from scripts.analyze_opes_dataset import main as opes_main


def main() -> int:
    opes_config = ROOT / "example-data" / "Ree_Data" / "E_end_end_distance_opes" / "analysis.toml"
    imetad_config = ROOT / "example-data" / "Ree_Data" / "E_end_end_distance_wt" / "analysis.toml"

    opes_main(["--config", str(opes_config)])
    imetad_main(["--config", str(imetad_config)])

    output_root = ROOT / "example-data" / "toml-example-results"
    manifest = {
        "description": "Ree example analysis driven entirely by TOML config files.",
        "opes_config": str(opes_config),
        "imetad_config": str(imetad_config),
        "outputs": {
            "opes": str(output_root / "opes"),
            "imetad": str(output_root / "imetad"),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
