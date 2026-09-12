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
        summary = dict(run_id=directory.name, scheduler_mode="paper_strict", paper_objective_scale_mode=mode,
                       seed=2025, NP=50, Ngen_max_stage1=200, Ngen_max_stage2=200,
                       stage1_initial_conflict_points=130, stage1_final_conflict_points=98, stage2_final_conflict_points=30,
                       final_Tdelay=100, final_Tair=1000, final_ORISK=20,
                       final_n_delay=1, changed_flight_count_two_stage=40,
                       final_paper_fitness=100, total_runtime=10)
        pd.DataFrame([summary]).to_csv(directory / "metrics_summary.csv", index=False)
        manifest=dict(run_id=directory.name,status="completed",objective_scale_mode=mode,quick=False)
        (directory / "run_manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
        for name in ("initial_plans.pkl", "risk_map.npy", "key_flights.csv"):
            (directory / name).write_bytes(b"identical input")
        pd.DataFrame(dict(plan_a=[0], plan_b=[1])).to_csv(directory / "conflicts_uncertain.csv", index=False)
        pd.DataFrame(dict(flight_id=[0])).to_csv(directory / "paper_stage1_strategy_assignment.csv", index=False)
    return directories


def test_scale_comparison_preserves_both_modes(completed_runs, tmp_path):
    frame = write_scale_comparison(*completed_runs, tmp_path,"cmp_test")
    assert frame.columns.tolist() == ["run_id", "comparison_id", "objective_mode", "initial_Nc", "stage1_Nc", "final_Nc", "Tdelay", "Tair", "ORISK", "n_delay", "changed_flights", "fitness", "runtime"]
    assert frame["objective_mode"].tolist() == ["raw_equation", "initial_reference_experimental"]
    output=tmp_path/"comparisons"/"cmp_test"
    assert (output/"objective_scale_comparison.csv").exists() and (output/"comparison_manifest.json").exists()


def test_scale_comparison_rejects_changed_input(completed_runs, tmp_path):
    (completed_runs[1] / "initial_plans.pkl").write_bytes(b"different input")
    with pytest.raises(ValueError, match="Initial inputs differ"):
        write_scale_comparison(*completed_runs, tmp_path)


def test_scale_comparison_rejects_quick_budget(completed_runs, tmp_path):
    path = completed_runs[1] / "metrics_summary.csv"
    frame = pd.read_csv(path)
    frame["NP"] = 20
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="budget does not match"):
        write_scale_comparison(*completed_runs, tmp_path)


def test_scale_comparison_does_not_overwrite(completed_runs,tmp_path):
    write_scale_comparison(*completed_runs,tmp_path,"cmp_fixed")
    with pytest.raises(FileExistsError):
        write_scale_comparison(*completed_runs,tmp_path,"cmp_fixed")


def test_scale_comparison_ignores_csv_run_id_only(completed_runs,tmp_path):
    for directory in completed_runs:
        path=directory/"conflicts_uncertain.csv"
        frame=pd.read_csv(path)
        frame.insert(0,"run_id",directory.name)
        frame.to_csv(path,index=False)
    frame=write_scale_comparison(*completed_runs,tmp_path,"cmp_ids")
    assert frame.run_id.tolist()==["raw","experimental"]
