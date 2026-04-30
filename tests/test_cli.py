from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from importlib.util import find_spec
from pathlib import Path


def _write_colvar(path: Path, rows) -> None:
    path.write_text(
        "\n".join(" ".join(str(value) for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


class CliTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy are required")
    def test_rates_cli_writes_json_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            colvar1 = tmp_path / "traj1.colvar"
            colvar2 = tmp_path / "traj2.colvar"
            output = tmp_path / "rates.json"
            repo_root = Path(__file__).resolve().parents[1]

            _write_colvar(colvar1, [(0, 0, 0.5), (1, 0, 0.5), (2, 0, 0.5)])
            _write_colvar(colvar2, [(0, 0, 0.2), (1, 0, 0.2), (2, 0, 0.2)])

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "eatr_rates",
                    "-i",
                    str(colvar1),
                    str(colvar2),
                    "-m",
                    "-q",
                    "-o",
                    str(output),
                ],
                cwd=tmp_path,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("iMetaD MLE ln k", payload)
            self.assertIn("iMetaD MLE KS stat", payload)

    @unittest.skipUnless(find_spec("numpy") and find_spec("scipy"), "numpy and scipy are required")
    def test_check_order_cli_writes_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output = tmp_path / "order.dat"
            repo_root = Path(__file__).resolve().parents[1]

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "check_order",
                    "-i",
                    "a.colvar",
                    "b.colvar",
                    "-l",
                    "a.log",
                    "b.log",
                    "-o",
                    str(output),
                ],
                cwd=tmp_path,
                env={**os.environ, "PYTHONPATH": str(repo_root)},
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                ["a.colvar | a.log", "b.colvar | b.log"],
            )
