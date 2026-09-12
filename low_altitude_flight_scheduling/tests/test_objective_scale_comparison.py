from __future__ import annotations

import json

import pandas as pd
import pytest

from compare_objective_scales import write_scale_comparison


@pytest.fixture
def completed_runs(tmp_path):
    directories = [tmp_path / "raw", tmp_path / "experimental"]
    for directory, mode in zip(directories, ("raw_equation", "initial_reference_experimental")):
        directory.mkdir()
        config = {"optimization": {"paper_objective_scale_mode": mode}, "seed": 2025}
        (directory / "paper_run_config.json").write_text(json.dumps(config), encoding="utf-8")
        summary = dict(scheduler_mode="paper_strict", paper_objective_scale_mode=mode,
                       seed=2025, NP=50, Ngen_max_stage1=200, Ngen_max_stage2=200,
                       stage1_final_conflict_points=98, stage2_final_conflict_points=30,
                       final_Tdelay=100, final_Tair=1000, final_ORISK=20,
                       final_n_delay=1, changed_flight_count_two_stage=40,
                       final_paper_fitness=100, total_runtime=10)
        pd.DataFrame([summary]).to_csv(directory / "metrics_summary.csv", index=False)
        for name in ("initial_plans.pkl", "risk_map.npy", "key_flights.csv"):
            (directory / name).write_bytes(b"identical input")
        pd.DataFrame(dict(plan_a=[0], plan_b=[1])).to_csv(directory / "conflicts_uncertain.csv", index=False)
        pd.DataFrame(dict(flight_id=[0])).to_csv(directory / "paper_stage1_strategy_assignment.csv", index=False)
    return directories


def test_scale_comparison_preserves_both_modes(completed_runs, tmp_path):
    output = tmp_path / "comparison.csv"
    frame = write_scale_comparison(*completed_runs, output)
    assert frame.columns.tolist() == ["mode", "stage1_Nc", "final_Nc", "Tdelay", "Tair", "ORISK", "n_delay", "changed_flights", "fitness", "runtime"]
    assert frame["mode"].tolist() == ["raw_equation", "initial_reference_experimental"]
    assert output.exists() and output.with_suffix(".md").exists()


def test_scale_comparison_rejects_changed_input(completed_runs, tmp_path):
    (completed_runs[1] / "initial_plans.pkl").write_bytes(b"different input")
    with pytest.raises(ValueError, match="Initial inputs differ"):
        write_scale_comparison(*completed_runs, tmp_path / "comparison.csv")


def test_scale_comparison_rejects_quick_budget(completed_runs, tmp_path):
    path = completed_runs[1] / "metrics_summary.csv"
    frame = pd.read_csv(path)
    frame["NP"] = 20
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="normal 50/200/200"):
        write_scale_comparison(*completed_runs, tmp_path / "comparison.csv")
