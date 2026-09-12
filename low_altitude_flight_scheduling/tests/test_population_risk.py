import copy
import math

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.grid import AirspaceGrid
from src.risk_map import (PopulationCenter, calculate_raw_risk, detect_population_centers_from_buildings,
                          generate_population_density, generate_risk_map, gravity_population_density,
                          population_density_per_m2)


def empty_grid(shape=(21, 2, 4)):
    return AirspaceGrid(shape=shape, obstacles=np.zeros(shape, dtype=bool))


def center(x=50, y=50, strength=10000, center_id=1):
    return PopulationCenter(center_id, x/100-0.5, y/100-0.5, x, y, 0.5, strength)


def test_population_units_and_raw_risk_linearity():
    assert float(population_density_per_m2(3500)) == pytest.approx(0.0035)
    cfg = load_config('nonexistent.yaml')
    grid = empty_grid()
    density = np.full(grid.shape[:2], 3500.0)
    raw = calculate_raw_risk(grid, cfg, density)
    assert np.allclose(calculate_raw_risk(grid, cfg, 2*density), 2*raw, rtol=1e-14, atol=0)
    # Audit the first layer's existing severity recipe and the m² * persons/m² boundary.
    h, m, g = 15, cfg['risk']['m'], 9.81
    terminal = math.sqrt(2*m*g/(1.225*cfg['risk']['C_D_b']))
    v_bal = terminal*math.sqrt(1-math.exp(-h/terminal))
    energy_bal = 0.5*m*v_bal**2
    t_par = math.sqrt(2*m*h/(cfg['risk']['A_p']*cfg['risk']['C_D_p']*g))
    energy_par = 0.5*m*max(2.0, h/t_par)**2
    severity = lambda e: 1/(1+100*(100/e)**(1/(4*cfg['risk']['S'])))
    area_m2 = math.pi*(cfg['risk']['r_UAV']+cfg['risk']['r_buf'])**2
    drift = 1+min(0.8, cfg['risk']['v_wind']*t_par/6000)
    expected = cfg['risk']['failure_probability']*area_m2*0.0035*(severity(energy_bal)+drift*severity(energy_par))
    assert raw[0, 0, 0] == pytest.approx(expected, rel=1e-13)


@pytest.mark.parametrize('value', [-1, np.nan, np.inf])
def test_invalid_population_rejected(value):
    with pytest.raises(ValueError):
        population_density_per_m2(value)


def test_exact_gravity_boundary_and_buildings_do_not_zero_population():
    grid = empty_grid()
    density = gravity_population_density(grid, [center()])
    assert density[0, 0] == pytest.approx(10000*math.e)
    assert density[9, 0] == pytest.approx(10000*math.exp(1-0.9**2))
    assert density[10, 0] == 3500  # r=1km is outside, not the interior formula.
    assert density[11, 0] == 3500
    grid.obstacles[0, 0, :] = True
    assert np.array_equal(gravity_population_density(grid, [center()]), density)


def test_nearest_center_only_not_sum_or_strongest_center():
    grid = empty_grid()
    density = gravity_population_density(grid, [center(), center(950, strength=90000, center_id=2)])
    assert density[0, 0] == pytest.approx(10000*math.e)
    assert density[9, 0] == pytest.approx(90000*math.e)
    assert gravity_population_density(grid, [])[0, 0] == 3500


@pytest.mark.parametrize('beta', [1.0, 2.0, 4.0])
def test_building_footprint_not_height_controls_centers(beta):
    city = AirspaceGrid()
    footprint = city.obstacles.any(axis=2)
    short = AirspaceGrid(obstacles=np.repeat(footprint[:, :, None], 4, axis=2))
    short.obstacles[:, :, 1:] = False
    tall = AirspaceGrid(obstacles=np.repeat(footprint[:, :, None], 4, axis=2))
    a = detect_population_centers_from_buildings(short, beta=beta)
    b = detect_population_centers_from_buildings(tall, beta=beta)
    assert a == b
    assert len(a) == 4
    assert max(c.center_population_density for c in a) == pytest.approx(3500*(1+beta))
    assert all(math.hypot(p.physical_x_m-q.physical_x_m, p.physical_y_m-q.physical_y_m) >= 1000
               for i, p in enumerate(a) for q in a[i+1:])
    assert a == detect_population_centers_from_buildings(short, beta=beta)


def test_static_population_independent_of_traffic_seed():
    a = load_config('nonexistent.yaml', {'environment_seed': 2025, 'traffic_seed': 2025})
    b = load_config('nonexistent.yaml', {'environment_seed': 2025, 'traffic_seed': 9876})
    ga, gb = AirspaceGrid.from_config(a), AirspaceGrid.from_config(b)
    assert np.array_equal(ga.obstacles, gb.obstacles)
    assert np.array_equal(generate_population_density(ga, a), generate_population_density(gb, b))
    assert np.array_equal(generate_risk_map(ga, a), generate_risk_map(gb, b))


def test_gaussian_guard_legacy_and_explicit_baseline():
    strict = load_config('nonexistent.yaml', {'population_model': 'old_gaussian'})
    city = AirspaceGrid()
    with pytest.raises(ValueError, match='Gaussian population is forbidden'):
        generate_population_density(city, strict)
    legacy = load_config('nonexistent.yaml', {'optimization': {'scheduler_mode': 'legacy_engineering'}})
    assert legacy['population_model'] == 'old_gaussian'
    baseline = copy.deepcopy(strict)
    baseline['experiment'] = {'network_only': True, 'allow_old_gaussian_baseline': True}
    assert np.array_equal(generate_population_density(city, baseline), generate_population_density(city, legacy))


def test_strict_baseline_and_static_mode_cannot_change_silently():
    cfg = load_config('nonexistent.yaml')
    city = AirspaceGrid()
    cfg['risk']['population_density_base'] = 10000
    with pytest.raises(ValueError, match='fixed at 3500'):
        generate_population_density(city, cfg)
    cfg['risk']['population_density_base'] = 3500
    cfg['population_map_mode'] = 'time_varying'
    with pytest.raises(ValueError, match='static_snapshot'):
        generate_population_density(city, cfg)


def test_population_and_risk_diagnostic_outputs(tmp_path):
    cfg = load_config('nonexistent.yaml')
    cfg['run'] = {'run_id': 'unit_population'}
    city = AirspaceGrid()
    risk = generate_risk_map(city, cfg, tmp_path)
    pop = np.load(tmp_path/'population_map.npy')
    assert pop.min() == 3500
    assert np.all(pop[city.obstacles.any(axis=2)] >= 3500)
    centers = pd.read_csv(tmp_path/'population_centers.csv')
    assert len(centers) == 4 and centers.run_id.eq('unit_population').all()
    assert centers.columns.tolist() == ['run_id', 'center_id', 'grid_x', 'grid_y', 'physical_x_m',
                                       'physical_y_m', 'building_density', 'center_population_density']
    stats = pd.read_csv(tmp_path/'risk_map_stats.csv')
    assert stats.level.tolist() == ['ground', 'z1', 'z2', 'z3', 'z4']
    for k in range(4):
        assert stats.iloc[k+1]['mean'] == pytest.approx(risk[:, :, k].mean())
    assert {'min', 'mean', 'median', 'max', 'std', 'p90', 'p95', 'p99'} <= set(stats)
    for name in ['population_density_map.png', 'risk_map_ground.png', 'risk_map_z1.png', 'risk_map_z2.png',
                 'risk_map_z3.png', 'risk_map_z4.png', 'risk_map_raw.npy', 'implementation_assumptions.md']:
        assert (tmp_path/name).exists()
    raw = np.load(tmp_path/'risk_map_raw.npy')
    assert np.allclose(raw, calculate_raw_risk(city, cfg, pop), atol=0)
    assert not np.all(risk[city.obstacles] == 1)  # Occupancy is a separate layer in strict mode.
