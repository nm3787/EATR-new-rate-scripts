from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_nnp_opes import main

DEFAULT_REE_CONFIG = ROOT / "example-data" / "Ree_Data" / "E_end_end_distance_opes" / "analysis.toml"


if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv:
        argv = ["--config", str(DEFAULT_REE_CONFIG)]
    raise SystemExit(main(argv))
