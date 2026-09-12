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
    run_id=(tmp_path / "latest_run.txt").read_text()
    output=tmp_path / "runs" / run_id
    assert (output / "metrics_summary.csv").exists()
    assert (output / "paper_convergence.csv").exists()
    assert (output / "paper_strict_implementation_report.md").exists()
    assert (output / "fata_run_report.html").exists()
    history = pd.read_csv(output / "paper_flight_strategy_history.csv")
    assert len(history) == 18
    for strategy in ("schedule", "speed", "reroute"):
        assert (history[f"final_{strategy}"] == (history[f"stage1_{strategy}"] | history[f"stage2_{strategy}"])).all()
    metrics = pd.read_csv(output / "metrics_summary.csv").iloc[0]
    assert metrics["paper_objective_scale_mode"] == "raw_equation"
    assert metrics["advanced_flight_count"] + metrics["delayed_flight_count"] <= 18
    for filename in output.rglob("*.csv"):
        frame=pd.read_csv(filename)
        assert frame.columns[0]=="run_id" and frame.run_id.eq(run_id).all()
    assert "Run completed" in completed.stdout and "Run ID:" in completed.stdout
    rebuilt=subprocess.run([sys.executable,"make_fata_report.py","--outputs",str(tmp_path),"--run-id",run_id],
                           cwd=root,check=True,capture_output=True,text=True,timeout=30)
    assert run_id in rebuilt.stdout and run_id in (output/"fata_run_report.html").read_text(encoding="utf-8")
