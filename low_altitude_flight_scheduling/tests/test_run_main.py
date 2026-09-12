from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_run_main_quick_generates_metrics(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "run_main.py", "--quick", "--seed", "2025", "--n-flights", "18", "--outputs", str(tmp_path)]
    completed = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True, timeout=180)
    assert "Main metrics summary" in completed.stdout
    assert (tmp_path / "metrics_summary.csv").exists()
    assert (tmp_path / "paper_convergence.csv").exists()
    assert (tmp_path / "paper_strict_implementation_report.md").exists()
    assert (tmp_path / "fata_run_report.html").exists()
