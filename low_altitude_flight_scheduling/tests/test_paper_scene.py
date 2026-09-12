import json

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.conflict_detection import Conflict
from src.conflict_network import collective_influence, build_conflict_network, select_paper_key_flights
from src.flight_plan import FlightPlan, generate_paper_random_flight_tasks
from src.grid import AirspaceGrid
from src.scene_diagnostics import analyze_initial_conflict_network, calibration_score, write_calibration_report


def test_paper_random_fixed_conditions_and_reproducibility():
    cfg = load_config("nonexistent.yaml")
    grid = AirspaceGrid(obstacles=np.zeros((60, 60, 4), dtype=bool))
    risk = np.zeros(grid.shape)
    plans = generate_paper_random_flight_tasks(grid, risk, cfg, n_flights=12, seed=2025)
    other = generate_paper_random_flight_tasks(grid, risk, cfg, n_flights=12, seed=2025)
    assert [(p.path, p.etd) for p in plans] == [(p.path, p.etd) for p in other]
    for p in plans:
        assert p.start[2] == p.goal[2] == p.path[0][2] == p.path[-1][2] == 0
        distance = np.linalg.norm((np.array(p.start) - p.goal) * grid.cell_size)
        assert 4800 <= distance <= 7200
        assert 0 <= p.etd <= 1800
        assert set(p.speed_profile) == {10.0}
        assert p.waypoint is None
    assert cfg['flight_generation']['mode'] == 'paper_random'
    assert cfg['conflict']['t_conflict'] == 30
    assert cfg['conflict']['sigma0'] == 1.0
    assert cfg['conflict']['sigma_rate'] == 0.01
    legacy = load_config('nonexistent.yaml', {'optimization': {'scheduler_mode': 'legacy_engineering'}})
    assert legacy['flight_generation']['mode'] == 'heterogeneous_random'
    assert legacy['conflict']['t_conflict'] == 20


@pytest.mark.parametrize(("ratio", "expected"), [(0.03, 3), (0.10, 10), (0.12, 12)])
def test_paper_ci_ratio_uses_important_ratio_only(ratio, expected):
    metrics = pd.DataFrame({"flight_id": range(100), "collective_influence": range(100)})
    assert len(select_paper_key_flights(metrics, 100, ratio)) == expected


def test_diagnostics_counts_events_and_unique_pair_cells(tmp_path):
    path = [(0, 0, 0), (1, 0, 0)]
    plans = [FlightPlan(i, path[0], path[-1], path, 0, [0, 10], [10], 0, 10) for i in range(12)]
    conflicts = [Conflict(0, i, (1, 0, 0), 1, 1, 10, 10, 0, 30, 'uncertain') for i in range(1, 12)]
    conflicts.append(Conflict(0, 1, (1, 0, 0), 2, 3, 20, 20, 0, 30, 'uncertain'))
    row, sensitivity = analyze_initial_conflict_network(plans, conflicts, [], 2025, 'test', tmp_path)
    assert row['raw_conflict_events'] == 12
    assert row['unique_spatial_conflict_points'] == 11
    assert row['same_pair_same_cell_multiple_index_count'] == 1
    assert row['network_edges'] == 11
    assert row['top1_conflict_point_share'] == 1
    assert row['top10_ci_point_coverage'] == 1
    json.dumps(row)
    assert len(sensitivity) == 3
    graph = build_conflict_network(plans, conflicts)
    before = collective_influence(graph, 2)
    for _, _, attrs in graph.edges(data=True):
        attrs['count'] = 10000
    assert collective_influence(graph, 2) == before
    metrics = pd.DataFrame({'flight_id': list(graph), 'collective_influence': list(before.values())})
    assert len(select_paper_key_flights(metrics, 100)) == 10
    assert calibration_score(dict(conflict_points_deterministic=53, conflict_points_uncertain=97,
                                 top10_ci_point_coverage=78/97)) == 0


def test_sampling_failure_does_not_relax_distance():
    cfg = load_config('nonexistent.yaml')
    cfg['flight_generation']['max_sampling_attempts'] = 10
    grid = AirspaceGrid(shape=(3, 3, 1), obstacles=np.zeros((3, 3, 1), dtype=bool))
    with pytest.raises(RuntimeError, match='tolerance unchanged'):
        generate_paper_random_flight_tasks(grid, np.zeros(grid.shape), cfg, n_flights=1)


def test_calibration_ignores_final_results_and_records_selection(tmp_path):
    good = dict(seed=2025, conflict_points_deterministic=53, conflict_points_uncertain=97,
                top10_ci_point_coverage=78/97, final_conflicts=1000, runtime=1000)
    bad = dict(seed=2026, conflict_points_deterministic=100, conflict_points_uncertain=150,
               top10_ci_point_coverage=0.2, final_conflicts=0, runtime=0.1)
    assert write_calibration_report(tmp_path, [bad, good], 2025, 2026)['seed'] == 2025
    report = (tmp_path / 'scene_calibration_report.md').read_text()
    assert 'Best seed: 2025' in report
    assert 'original random seed and full simulation environment were not disclosed' in report
