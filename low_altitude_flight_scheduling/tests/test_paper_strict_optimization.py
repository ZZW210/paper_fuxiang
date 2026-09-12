from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from src import fata, optimization_model
from src.adm_matching import (initialize_probability_matrix, sample_strategy_species,
                             update_probability_matrix)
from src.config import load_config, apply_quick_overrides
from src.conflict_detection import detect_conflicts
from src.conflict_network import select_paper_key_flights
from src.flight_plan import FlightPlan, compute_eta_times
from src.grid import AirspaceGrid
from src.paper_optimization import (PaperReference, paper_atd_bounds, paper_objective_components,
    paper_fitness, conflict_weight_delta, build_stage1_decision_layout, build_stage2_decision_layout,
    decode_stage1_solution, decode_stage2_solution, InfeasiblePaperRoute,
    decode_activation_genes, flight_strategies, stage2_flight_strategies)
from src.paper_scheduler import optimize_paper_schedule


@pytest.fixture
def scenario():
    cfg = load_config("nonexistent.yaml")
    cfg["optimization"]["n_jobs"] = 1
    cfg["fata"].update(NP=4, Ngen_max_stage1=2, Ngen_max_stage2=2)
    grid = AirspaceGrid(shape=(8, 8, 4), obstacles=np.zeros((8, 8, 4), dtype=bool))
    path = [(i, 3, 1) for i in range(1, 7)]
    plans = [FlightPlan(i, path[0], path[-1], list(path), 1000.0,
                       compute_eta_times(path, 1000.0, 10.0, grid.cell_size),
                       [10.0] * (len(path) - 1), float(len(path)), 50.0) for i in range(2)]
    risk = np.ones(grid.shape)
    conflicts = detect_conflicts(plans, cfg)
    return cfg, grid, risk, plans, conflicts


def _vector(layout):
    return (layout.lower + layout.upper) / 2


@pytest.mark.parametrize("atd", [800.0, 1200.0])
def test_atd_early_and_late_absolute_delay(scenario, atd):
    cfg, grid, risk, plans, conflicts = scenario
    layout = build_stage1_decision_layout(plans, [0], conflicts, cfg, grid)
    vector = _vector(layout)
    vector[layout.blocks[0].activation_genes] = [1, 0, 0]
    vector[layout.blocks[0].atd_gene] = atd
    decoded = decode_stage1_solution(vector, plans, layout, cfg, grid, risk)
    assert decoded[0].atd == atd
    assert decoded[0].speed_profile == plans[0].speed_profile
    components = paper_objective_components(decoded, plans, risk, [], cfg)
    assert components["Tdelay"] == 200.0
    assert components["n_delay"] == int(atd > 1000.0)
    assert plans[0].etd == 1000.0


@pytest.mark.parametrize("etd, expected", [(1, (1, 1801)), (1000, (1, 2800)), (3500, (1700, 3600))])
def test_atd_bounds(scenario, etd, expected):
    cfg, _, _, plans, _ = scenario
    plan = plans[0].copy()
    plan.etd = etd
    assert paper_atd_bounds(plan, cfg) == expected
    for value in np.linspace(*expected, 21):
        assert 1 <= value <= 3600 and abs(etd - value) <= 1800


def test_continuous_per_segment_speed(scenario):
    cfg, grid, risk, plans, conflicts = scenario
    layout = build_stage1_decision_layout(plans, [0], conflicts, cfg, grid)
    vector = _vector(layout)
    block = layout.blocks[0]
    vector[block.activation_genes] = [0, 1, 0]
    vector[block.speed_genes] = 10.0
    vector[block.speed_genes.start + 2] = 11.374
    vector[block.atd_gene] = 700.0  # Ignored for a speed-only individual.
    decoded = decode_stage1_solution(vector, plans, layout, cfg, grid, risk)[0]
    assert decoded.speed_profile[2] == 11.374
    assert decoded.atd == plans[0].etd
    assert decoded.eta_times[:3] == plans[0].eta_times[:3]
    assert decoded.eta_times[3] == pytest.approx(decoded.eta_times[2] + 100 / 11.374)
    assert all(a < b for a, b in zip(decoded.eta_times[3:], plans[0].eta_times[3:]))


def test_reroute_optimized_via_and_feasibility(scenario):
    cfg, grid, risk, plans, conflicts = scenario
    layout = build_stage1_decision_layout(plans, [0], conflicts, cfg, grid)
    vector = _vector(layout)
    block = layout.blocks[0]
    vector[block.activation_genes] = [0, 0, 1]
    vector[block.atd_gene] = 700.0
    vector[block.speed_genes] = 19.123
    for genes in block.reroute_genes:
        vector[genes] = [4.2, 5.1, 2.1]
    decoded = decode_stage1_solution(vector, plans, layout, cfg, grid, risk)[0]
    assert (4, 5, 2) in decoded.path
    assert decoded.path[0] == plans[0].start and decoded.path[-1] == plans[0].goal
    assert decoded.atd == 1000.0
    assert all(v == 10.0 for v in decoded.speed_profile)
    assert len(decoded.speed_profile) == len(decoded.path) - 1
    assert all(grid.is_free(c) for c in decoded.path)
    assert all(max(abs(a[i] - b[i]) for i in range(3)) <= 1 and a != b for a, b in zip(decoded.path, decoded.path[1:]))
    assert decoded.risk_sum == len(decoded.path)
    assert decoded.total_air_time == decoded.eta_times[-1] - decoded.atd
    grid.obstacles[4, 5, 2] = True
    with pytest.raises(InfeasiblePaperRoute):
        decode_stage1_solution(vector, plans, layout, cfg, grid, risk)


def test_eq49_to_51_exact_and_no_hard_conflict_penalty(scenario):
    cfg, _, _, _, _ = scenario
    reference = PaperReference(3600.0, 100.0, 12.0, 6.0)
    components = dict(Tdelay=200.0, Tair=120.0, ORISK=15.0, Nc=3, n_delay=2, n_battery=1)
    expected_obj = 0.2 * (0.25 * 200 + 0.25 * 120 + 0.5 * 15) + 0.8 * 0.9 * 3 * 120
    assert paper_fitness(components, reference, cfg, 200, 200) == pytest.approx(expected_obj + 1000 + 200)
    components.update(n_delay=0, n_battery=0)
    assert paper_fitness(components, reference, cfg, 200, 200) == pytest.approx(expected_obj)
    cfg["optimization"].update(conflict_hard_penalty=1e20, delay_count_cap=0, max_changed_flight_ratio=0)
    assert paper_fitness(components, reference, cfg, 200, 200) == pytest.approx(expected_obj)
    assert conflict_weight_delta(200, 200) == pytest.approx(0.9)
    assert conflict_weight_delta(0, 200) == 1.0
    cfg["optimization"]["paper_objective_scale_mode"] = "initial_reference_experimental"
    expected_norm = 0.2 * (0.25 * 200 / 3600 + 0.25 * 1.2 + 0.5 * 15 / 12) + 0.8 * 0.9 * 0.5 * 1.2
    assert paper_fitness(components, reference, cfg, 200, 200) == pytest.approx(expected_norm)
    cfg["optimization"]["paper_objective_scale_mode"] = "unknown"
    with pytest.raises(ValueError):
        paper_fitness(components, reference, cfg, 200, 200)


def test_delay_epsilon_and_battery(scenario):
    cfg, _, risk, plans, _ = scenario
    final = [p.copy() for p in plans]
    final[0].atd += 1e-7
    final[1].atd += 1e-5
    final[0].eta_times[-1] = final[0].atd + 1200.0
    final[1].eta_times[-1] = final[1].atd + 1200.01
    components = paper_objective_components(final, plans, risk, [], cfg)
    assert components["n_delay"] == 1
    assert components["n_battery"] == 1


def test_fixed_unweighted_ci_top_ten():
    metrics = pd.DataFrame(dict(flight_id=range(100), collective_influence=range(100),
                                weighted_collective_influence=range(100, 0, -1)))
    assert select_paper_key_flights(metrics, 100) == list(range(99, 89, -1))


def test_uniform_adm_independent_sampling_and_update():
    probability = initialize_probability_matrix(2)
    assert np.array_equal(probability, np.full((2, 3), 1 / 3))
    sampled = sample_strategy_species(probability, 3000, np.random.default_rng(1))
    assert len(np.unique(sampled, axis=0)) == 9
    dominant = np.array([[0, 1], [0, 2]])
    expected = 0.5 * probability + 0.5 * np.array([[1, 0, 0], [0, 0.5, 0.5]])
    assert np.allclose(update_probability_matrix(probability, dominant), expected)


@pytest.mark.parametrize("mask", range(1, 8))
def test_seven_strategy_combinations(scenario, mask):
    cfg, grid, risk, plans, conflicts = scenario
    active = tuple(i for i in range(3) if mask & (1 << i))
    layout = build_stage1_decision_layout(plans, [0], conflicts, cfg, grid)
    vector = _vector(layout)
    block = layout.blocks[0]
    genes = [float(i in active) for i in range(3)]
    assert decode_activation_genes(genes) == active
    vector[block.activation_genes] = genes
    vector[block.atd_gene] = 800
    vector[block.speed_genes] = 11.374
    for route in block.reroute_genes:
        vector[route] = [4, 5, 2]
    plan = decode_stage1_solution(vector, plans, layout, cfg, grid, risk)[0]
    assert flight_strategies(plan) == active
    assert plan.atd == (800 if 0 in active else 1000)
    assert all(v == (11.374 if 1 in active else 10) for v in plan.speed_profile)
    assert plan.rerouted == (2 in active)
    assert decode_activation_genes([0.1, 0.3, 0.2]) == (1,)


def test_stage2_retains_stage1_and_opens_both_endpoints(scenario):
    cfg, grid, risk, plans, conflicts = scenario
    stage1 = [p.copy() for p in plans]
    stage1[0].etd = 990.0
    stage1[0].delay = -10.0
    stage1[0].paper_strategies = (0, 1)
    stage1[0].speed_profile = [11.374] * 5
    stage1[0].eta_times = compute_eta_times(stage1[0].path, 990, stage1[0].speed_profile, grid.cell_size)
    conflicts = detect_conflicts(stage1, cfg)
    layout = build_stage2_decision_layout(stage1, conflicts, cfg, grid, plans)
    vector = _vector(layout)
    for block in layout.blocks:
        vector[block.reroute_enable_genes] = 0
    species = np.full(len(conflicts), 2)
    assert stage2_flight_strategies(layout, species) == {0: (2,), 1: (2,)}
    decoded = decode_stage2_solution(vector, stage1, plans, layout, species, cfg, grid, risk)
    assert decoded[0].atd == 990 and decoded[0].delay == -10
    assert decoded[0].speed_profile == stage1[0].speed_profile
    assert decoded[0].path == stage1[0].path
    assert flight_strategies(decoded[0]) == (0, 1, 2)
    assert decoded[1].path == plans[1].path and not decoded[1].changed
    mixed = np.arange(len(conflicts)) % 3
    original = mixed.copy()
    assert stage2_flight_strategies(layout, mixed) == {0: (0, 1, 2), 1: (0, 1, 2)}
    assert np.array_equal(original, mixed)


def test_stage2_can_modify_one_both_or_neither_endpoint(scenario):
    cfg, grid, risk, plans, conflicts = scenario
    layout = build_stage2_decision_layout(plans, conflicts, cfg, grid, plans)
    species = np.zeros(len(conflicts), dtype=int)
    for selected in ((), (0,), (0, 1)):
        vector = _vector(layout)
        for block in layout.blocks:
            vector[block.atd_gene] = 800 if block.flight_id in selected else 1000
        decoded = decode_stage2_solution(vector, plans, plans, layout, species, cfg, grid, risk)
        assert tuple(p.id for p in decoded if p.changed) == selected


@pytest.mark.parametrize("strategy", [1, 2])
@pytest.mark.parametrize("selected", [(), (0,), (0, 1)])
def test_stage2_speed_and_reroute_allow_endpoint_noop(scenario, strategy, selected):
    cfg, grid, risk, plans, conflicts = scenario
    layout = build_stage2_decision_layout(plans, conflicts, cfg, grid, plans)
    vector = _vector(layout)
    for block in layout.blocks:
        vector[block.speed_genes] = 11.374 if block.flight_id in selected else 10
        vector[block.reroute_enable_genes] = float(block.flight_id in selected)
        for genes in block.reroute_genes:
            vector[genes] = [4, 5, 2]
    final = decode_stage2_solution(vector, plans, plans, layout, np.full(len(conflicts), strategy), cfg, grid, risk)
    assert tuple(p.id for p in final if p.changed) == selected


def test_stage2_layout_uses_rerouted_stage1_geometry(scenario):
    cfg, grid, risk, plans, conflicts = scenario
    layout1 = build_stage1_decision_layout(plans, [0, 1], conflicts, cfg, grid)
    vector = _vector(layout1)
    for block in layout1.blocks:
        vector[block.activation_genes] = [0, 1, 1]
        vector[block.speed_genes] = 11.374
        for genes in block.reroute_genes:
            vector[genes] = [4, 5, 2]
    current = decode_stage1_solution(vector, plans, layout1, cfg, grid, risk)
    remaining = detect_conflicts(current, cfg)
    assert any(c.cell not in plans[0].path for c in remaining)
    layout2 = build_stage2_decision_layout(current, remaining, cfg, grid, plans)
    noop = _vector(layout2)
    for block in layout2.blocks:
        noop[block.reroute_enable_genes] = 0
        assert max(block.segment_indices) == len(current[block.flight_id].path) - 2
    final = decode_stage2_solution(noop, current, plans, layout2, np.full(len(remaining), 2), cfg, grid, risk)
    assert [p.path for p in final] == [p.path for p in current]
    assert [p.speed_profile for p in final] == [p.speed_profile for p in current]
    assert all(p.changed for p in final)
    with pytest.raises(ValueError, match="stage input path"):
        build_stage2_decision_layout(plans, remaining, cfg, grid, plans)


def test_stage2_speed_only_opens_sampled_conflict_segments(scenario):
    cfg, grid, risk, plans, conflicts = scenario
    cfg["paper_encoding"]["local_window_segments"] = 1
    selected = [conflicts[0], conflicts[-1]]
    layout = build_stage2_decision_layout(plans, selected, cfg, grid, plans)
    vector = _vector(layout)
    for block in layout.blocks:
        vector[block.speed_genes] = 11.374
        vector[block.atd_gene] = 1000
    final = decode_stage2_solution(vector, plans, plans, layout, [1, 0], cfg, grid, risk)
    for block in layout.blocks:
        for index, speed in enumerate(final[block.flight_id].speed_profile):
            assert speed == (11.374 if index in block.conflict_segments[0] else 10)


def test_adm_dominance_keeps_original_sampled_labels(scenario, monkeypatch):
    from src import adm_matching as adm
    cfg, grid, risk, plans, conflicts = scenario
    layout = build_stage2_decision_layout(plans, conflicts, cfg, grid, plans)
    sampled = np.stack([np.arange(len(conflicts)) % 3, np.full(len(conflicts), 2),
                        np.full(len(conflicts), 1), np.full(len(conflicts), 0)])
    untouched = sampled.copy()
    monkeypatch.setattr(adm, "sample_strategy_species", lambda *args: sampled.copy())
    seen = []
    actual_update = adm.update_probability_matrix
    def tracked(probability, dominant, rate):
        seen.append(dominant.copy())
        return actual_update(probability, dominant, rate)
    monkeypatch.setattr(adm, "update_probability_matrix", tracked)
    def fake_fata(objective, lower, upper, dim, **kwargs):
        context = kwargs["generation_context"](1, 4, np.random.default_rng(1))
        assert np.array_equal(context, sampled)
        kwargs["on_generation_evaluated"](1, None, np.array([1, 3, 2, 4]), context)
        vector = (lower + upper) / 2
        for block in layout.blocks:
            vector[block.reroute_enable_genes] = 0
        return fata.PaperFATAResult(vector, 1, [1], [], [], context[0].copy())
    monkeypatch.setattr(fata, "fata_optimize_paper", fake_fata)
    reference = PaperReference.from_initial(plans, risk, conflicts)
    result = adm.adm_fata_optimize(plans, plans, conflicts, layout, cfg, grid, risk, reference)
    assert np.array_equal(seen[0], sampled[[0]])
    assert np.array_equal(result.best_strategies, sampled[0])
    assert np.array_equal(result.final_probability, actual_update(np.full((len(conflicts), 3), 1/3), sampled[[0]], .5))
    assert np.array_equal(sampled, untouched)


def test_strict_path_never_calls_repair_and_stage2_runs_fata(scenario, monkeypatch):
    cfg, grid, risk, plans, conflicts = scenario
    forbidden = ["greedy_time_deconfliction", "repair_conflicts", "try_apply_action_with_rollback",
                 "_delay_candidates", "_speed_factors", "independent_matching_deconfliction"]
    def fail(*args, **kwargs):
        raise AssertionError("Engineering function called by strict mode")
    for name in forbidden:
        monkeypatch.setattr(optimization_model, name, fail, raising=False)
    calls = []
    actual = fata.fata_optimize_paper
    def tracked(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            layout = build_stage1_decision_layout(plans, [0], conflicts, cfg, grid)
            vector = _vector(layout)
            vector[layout.blocks[0].activation_genes] = [1, 0, 0]
            vector[layout.blocks[0].atd_gene] = plans[0].etd
            score = args[0](vector, kwargs["max_iter"])
            return fata.PaperFATAResult(vector, score, [score], [], [], None)
        return actual(*args, **kwargs)
    monkeypatch.setattr(fata, "fata_optimize_paper", tracked)
    result = optimize_paper_schedule(plans, conflicts, [0], cfg, grid, risk, progress=False)
    assert len(calls) == 2
    assert calls[1]["objective_with_context"] is not None
    assert result.adm is not None and len(result.adm.final_probability) == len(conflicts)
    assert len(result.adm.probability_history) == cfg["fata"]["Ngen_max_stage2"] + 1


def test_good_point_initialization_exact_and_no_gaussian_search(monkeypatch):
    seen = []
    def objective(vector, generation):
        seen.append(vector.copy())
        return float(np.sum(vector * vector))
    lb, ub = np.array([-2.0, 1.0]), np.array([3.0, 8.0])
    fata.fata_optimize_paper(objective, lb, ub, 2, population=4, max_iter=1, n_jobs=1)
    assert np.array_equal(np.asarray(seen), fata.good_point_set(4, 2, lb, ub))
    source = inspect.getsource(fata.fata_optimize_paper)
    assert ".normal(" not in source and "local_count" not in source


def paper_sphere(vector, generation):
    return float(np.sum(vector ** 2) * conflict_weight_delta(generation, 4))


def test_parallel_fitness_reproducible():
    serial = fata.fata_optimize_paper(paper_sphere, -2, 2, 3, population=8, max_iter=4, n_jobs=1, seed=9)
    parallel = fata.fata_optimize_paper(paper_sphere, -2, 2, 3, population=8, max_iter=4, n_jobs=8, seed=9)
    assert np.array_equal(serial.best_position, parallel.best_position)
    assert serial.convergence == parallel.convergence


def test_real_adm_parallel_reproducible(scenario):
    from src.adm_matching import adm_fata_optimize

    cfg, grid, risk, plans, conflicts = scenario
    cfg["fata"]["NP"] = 8
    layout = build_stage2_decision_layout(plans, conflicts, cfg, grid)
    reference = PaperReference.from_initial(plans, risk, conflicts)
    serial = adm_fata_optimize(plans, plans, conflicts, layout, cfg, grid, risk, reference, seed=27)
    cfg["optimization"]["n_jobs"] = 8
    parallel = adm_fata_optimize(plans, plans, conflicts, layout, cfg, grid, risk, reference, seed=27)
    assert serial.convergence == parallel.convergence
    assert np.array_equal(serial.best_decision_vector, parallel.best_decision_vector)
    assert np.array_equal(serial.best_strategies, parallel.best_strategies)
    assert np.array_equal(serial.final_probability, parallel.final_probability)
    assert serial.best_flight_strategies == parallel.best_flight_strategies
    assert [p.path for p in serial.best_plans] == [p.path for p in parallel.best_plans]
    assert [p.eta_times for p in serial.best_plans] == [p.eta_times for p in parallel.best_plans]


def test_generation_delta_reaches_every_evaluation():
    seen = []
    def objective(vector, generation):
        seen.append(generation)
        return float(np.sum(vector ** 2) + conflict_weight_delta(generation, 4))
    result = fata.fata_optimize_paper(objective, 1, 2, 2, population=4, max_iter=4, n_jobs=1)
    assert set(seen) == {1, 2, 3, 4}
    assert result.best_fitness == objective(result.best_position, 4)


def test_no_remaining_conflicts_skips_stage2(scenario, monkeypatch):
    import src.paper_scheduler as scheduler
    cfg, grid, risk, plans, _ = scenario
    def fail(*args, **kwargs):
        raise AssertionError("ADM should be skipped")
    monkeypatch.setattr(scheduler, "adm_fata_optimize", fail)
    result = scheduler.optimize_paper_schedule([plans[0]], [], [0], cfg, grid, risk, progress=False)
    assert result.adm is None and result.final is result.stage1


def test_quick_only_changes_budget():
    cfg = load_config("nonexistent.yaml")
    quick = apply_quick_overrides(cfg)
    assert quick["optimization"] == cfg["optimization"]
    assert quick["adm"] == cfg["adm"] and quick["paper_encoding"] == cfg["paper_encoding"]
    assert quick["fata"]["NP"] == 20
    assert quick["fata"]["Ngen_max_stage1"] == quick["fata"]["Ngen_max_stage2"] == 50
