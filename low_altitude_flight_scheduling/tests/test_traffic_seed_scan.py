from __future__ import annotations

from scan_traffic_seeds import evaluate_traffic_seed, prepare_environment, run_seed_batch
from src.config import load_config, resolve_scene_seeds
from src.flight_plan import sample_paper_random_traffic


def _config():
    cfg = load_config("config.yaml", {"optimization": {"scheduler_mode": "paper_strict"}})
    resolve_scene_seeds(cfg, environment_seed=2025, traffic_seed=0)
    return cfg


def test_fixed_environment_hashes_do_not_depend_on_traffic_seed():
    cfg = _config()
    grid, risk, hashes = prepare_environment(cfg, 2025)
    first = evaluate_traffic_seed(cfg, grid, risk, hashes, 0, "test_scan")
    second = evaluate_traffic_seed(cfg, grid, risk, hashes, 1, "test_scan")
    assert {key: first[key] for key in hashes} == {key: second[key] for key in hashes}


def test_same_traffic_seed_reproduces_tasks_and_initial_metrics():
    cfg = _config()
    grid, risk, hashes = prepare_environment(cfg, 2025)
    assert sample_paper_random_traffic(grid, cfg, 100, 17) == sample_paper_random_traffic(grid, cfg, 100, 17)
    first = evaluate_traffic_seed(cfg, grid, risk, hashes, 17, "test_scan")
    second = evaluate_traffic_seed(cfg, grid, risk, hashes, 17, "test_scan")
    for key in ("det_Nc", "unc_Nc", "conflict_edges", "CI_top10_point_coverage", "calibration_score"):
        assert first[key] == second[key]


def test_single_and_parallel_seed_batches_match():
    cfg = _config()
    serial, _ = run_seed_batch(cfg, 2025, [3], 1, "test_scan")
    parallel, _ = run_seed_batch(cfg, 2025, [3], 2, "test_scan")
    for key in ("det_Nc", "unc_Nc", "conflict_edges", "CI_top10_point_coverage", "calibration_score"):
        assert serial[0][key] == parallel[0][key]
