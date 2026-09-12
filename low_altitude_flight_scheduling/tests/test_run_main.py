from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def test_run_main_quick_generates_metrics(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, "run_main.py", "--quick", "--seed", "2025", "--n-flights", "18", "--outputs", str(tmp_path)]
    completed = subprocess.run(cmd, cwd=root, check=True, capture_output=True, text=True, timeout=180)
    assert "Main metrics summary" in completed.stdout
    assert (tmp_path / "metrics_summary.csv").exists()
    assert (tmp_path / "paper_convergence.csv").exists()
    assert (tmp_path / "paper_strict_implementation_report.md").exists()
    assert (tmp_path / "fata_run_report.html").exists()
    history = pd.read_csv(tmp_path / "paper_flight_strategy_history.csv")
    assert len(history) == 18
    for strategy in ("schedule", "speed", "reroute"):
        assert (history[f"final_{strategy}"] == (history[f"stage1_{strategy}"] | history[f"stage2_{strategy}"])).all()
    metrics = pd.read_csv(tmp_path / "metrics_summary.csv").iloc[0]
    assert metrics["paper_objective_scale_mode"] == "raw_equation"
    assert metrics["advanced_flight_count"] + metrics["delayed_flight_count"] <= 18
