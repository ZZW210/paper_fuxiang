import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.astar_3d import astar_distance_unit_m
from src.config import load_config, resolve_scene_seeds
from src.flight_plan import (PaperTrafficTask, plan_paper_random_traffic, sample_paper_random_traffic, FlightPlan)
from src.grid import AirspaceGrid
from src.risk_route_network import run_structure_experiments, EXPERIMENT_VARIANTS
from src.run_archive import generate_network_run_id, validate_run_id
from src.scene_diagnostics import route_concentration_metrics


@pytest.mark.parametrize('mode,unit,move', [('meter', 1, 100), ('grid', 100, 1), ('kilometer', 1000, 0.1)])
def test_distance_scale_forwarding_preserves_risk_weights(monkeypatch, mode, unit, move):
    cfg = load_config('nonexistent.yaml')
    cfg['astar_distance_scale_mode'] = mode
    assert astar_distance_unit_m(mode) == unit
    assert 100/unit == move
    captured = {}
    def fake_astar(grid, start, goal, risk, **kwargs):
        captured.update(kwargs)
        return [start, goal]
    monkeypatch.setattr('src.flight_plan.astar_path', fake_astar)
    city = AirspaceGrid(shape=(3, 3, 1), obstacles=np.zeros((3, 3, 1), dtype=bool))
    plans = plan_paper_random_traffic(city, np.zeros(city.shape), cfg,
                                    [PaperTrafficTask(0, (0, 0, 0), (1, 0, 0), 123.0, 10.0)])
    assert captured == {'alpha_r': 0.8, 'alpha_l': 0.2, 'distance_unit_m': unit}
    assert plans[0].etd == 123 and plans[0].total_air_time == 10


def test_seed_precedence_and_frozen_traffic():
    cfg = load_config('nonexistent.yaml')
    assert resolve_scene_seeds(cfg, seed=111, traffic_seed=222) == (111, 222)
    assert cfg['environment_seed'] == 111 and cfg['traffic_seed'] == 222
    city = AirspaceGrid.from_config(cfg)
    tasks = sample_paper_random_traffic(city, cfg, 4)
    assert tasks == sample_paper_random_traffic(city, cfg, 4)
    cfg['astar_distance_scale_mode'] = 'kilometer'
    cfg['population']['beta'] = 4
    assert tasks == sample_paper_random_traffic(city, cfg, 4)
    assert tasks != sample_paper_random_traffic(city, cfg, 4, seed=223)
    assert load_config('nonexistent.yaml')['astar_distance_scale_mode'] == 'meter'
    with pytest.raises(ValueError):
        astar_distance_unit_m('automatic_best')


def test_failed_fixed_route_does_not_resample(monkeypatch):
    cfg = load_config('nonexistent.yaml')
    city = AirspaceGrid()
    task = sample_paper_random_traffic(city, cfg, 1)[0]
    monkeypatch.setattr('src.flight_plan.astar_path', lambda *args, **kwargs: [])
    with pytest.raises(RuntimeError, match='traffic and tolerance unchanged'):
        plan_paper_random_traffic(city, np.zeros(city.shape), cfg, [task])


def test_route_reuse_counts_distinct_flights_not_repeat_visits():
    city = AirspaceGrid(shape=(3, 3, 1), obstacles=np.zeros((3, 3, 1), dtype=bool))
    p = [(0, 0, 0), (1, 0, 0), (0, 0, 0)]
    plans = [FlightPlan(i, p[0], p[-1], p, 0, [0, 10, 20], [10, 10], 0, 20) for i in range(2)]
    metrics = route_concentration_metrics(plans, city)
    assert metrics['route_cell_reuse_max'] == 2
    assert metrics['route_cell_reuse_p90'] == 2
    assert metrics['route_visited_3d_cells'] == 2


def test_network_run_ids_include_both_seeds_and_are_unique():
    values = {generate_network_run_id('reference28_gravity', 'grid', 2025, 2030, 'abcdefghi') for _ in range(100)}
    assert len(values) == 100
    for value in values:
        validate_run_id(value)
        assert '_network_reference28_gravity_grid_env2025_traffic2030_abcdefgh' in value


def test_four_group_pipeline_freezes_inputs_and_never_overwrites(tmp_path):
    cfg = load_config('nonexistent.yaml')
    # Small traffic smoke test; still real grid, population, risk, A*, detector, network and CI.
    rows, comparison = run_structure_experiments(cfg, tmp_path, n_flights=2)
    assert [(r['population_mode'], r['distance_scale']) for r in rows] == list(EXPERIMENT_VARIANTS)
    assert cfg['population_model'] == 'reference28_gravity' and 'experiment' not in cfg
    inputs, populations, obstacle_hashes = [], [], []
    for row in rows:
        directory = tmp_path/'runs'/row['run_id']
        manifest = json.loads((directory/'run_manifest.json').read_text(encoding='utf-8'))
        assert manifest['status'] == 'completed' and manifest['optimizers_executed'] is False
        assert not {'Stage1', 'Stage2', 'FATA', 'ADM'} & set(manifest['stages_executed'])
        assert manifest['environment_seed'] == manifest['traffic_seed'] == 2025
        inputs.append(pd.read_csv(directory/'traffic_tasks.csv').drop(columns='run_id'))
        obstacle_hashes.append(manifest['obstacle_sha256'])
        populations.append(np.load(directory/'population_map.npy'))
        for filename in directory.glob('*.csv'):
            frame = pd.read_csv(filename)
            assert frame.columns[0] == 'run_id' and frame.run_id.eq(row['run_id']).all()
    for frame in inputs[1:]:
        pd.testing.assert_frame_equal(frame, inputs[0])
    assert len(set(obstacle_hashes)) == 1
    assert np.array_equal(populations[1], populations[2]) and np.array_equal(populations[2], populations[3])
    assert not np.array_equal(populations[0], populations[1])
    assert len(pd.read_csv(comparison/'risk_route_network_comparison.csv')) == 4
    report = (comparison/'risk_route_network_report.md').read_text(encoding='utf-8')
    assert '不能确认低 Top10 coverage 的主要成因已定位' in report
    old = (comparison/'risk_route_network_comparison.csv').read_bytes()
    # A collision fails before any old result is overwritten.
    from unittest.mock import patch
    with patch('src.risk_route_network.generate_network_run_id', return_value=comparison.name):
        with pytest.raises(FileExistsError):
            run_structure_experiments(cfg, tmp_path, n_flights=2)
    assert (comparison/'risk_route_network_comparison.csv').read_bytes() == old
