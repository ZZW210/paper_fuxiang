import copy

import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.conflict_detection import count_conflict_pairs, detect_conflicts
from src.flight_plan import FlightPlan, compute_eta_times
from src.grid import AirspaceGrid
from src.paper_optimization import (build_stage1_decision_layout, build_stage2_decision_layout,
                                    decode_stage1_solution, decode_stage2_solution, PaperPopulationObjective,
                                    PaperReference)
from src.stage2_rollback import acceptance


@pytest.mark.parametrize("before,after,accepted,reason", [
    ((3, 5), (2, 9), True, ""), ((3, 5), (3, 4), True, ""),
    ((3, 5), (4, 1), False, "pairs_increased"),
    ((3, 5), (3, 5), False, "same_pairs_points_not_decreased"),
    ((3, 5), (3, 6), False, "same_pairs_points_not_decreased")])
def test_strict_pair_point_rule(before, after, accepted, reason):
    assert acceptance(before, after) == (accepted, reason)


def scenario(tmp_path):
    cfg = load_config("nonexistent.yaml")
    cfg["optimization"].update(n_jobs=1, stage2_global_rollback=True, stage2_rollback_log_dir=str(tmp_path))
    grid = AirspaceGrid(shape=(8, 8, 4), obstacles=np.zeros((8, 8, 4), dtype=bool))
    path = [(i, 3, 1) for i in range(1, 7)]
    plans = [FlightPlan(i, path[0], path[-1], list(path), 1000.0,
                       compute_eta_times(path, 1000, 10, grid.cell_size), [10.0]*5, 6.0, 50.0) for i in range(2)]
    risk = np.ones(grid.shape)
    conflicts = detect_conflicts(plans, cfg)
    layout = build_stage2_decision_layout(plans, conflicts, cfg, grid, plans)
    x = (layout.lower+layout.upper)/2
    x[layout.actor_genes] = 0
    for block in layout.blocks:
        x[block.atd_gene] = 1000
        x[block.speed_genes] = 10
        for genes in block.reroute_genes:
            x[genes] = [4, 5, 2]
    return cfg, grid, risk, plans, conflicts, layout, x


def test_noop_restores_exact_reference_and_logs(tmp_path):
    cfg, grid, risk, plans, conflicts, layout, x = scenario(tmp_path)
    before = copy.deepcopy(plans)
    result = decode_stage2_solution(x, plans, plans, layout, np.zeros(len(conflicts), dtype=int), cfg, grid, risk)
    assert result[0] is plans[0] and result[1] is plans[1]
    assert plans == before
    logs = pd.read_csv(next(tmp_path.glob("actions_*.csv")))
    assert len(logs) == 1 and not logs.accepted.iloc[0]
    assert logs.rollback_reason.iloc[0] == "same_pairs_points_not_decreased"


@pytest.mark.parametrize("strategy", [0, 1, 2])
def test_full_detection_and_nonworsening_all_strategies(tmp_path, strategy, monkeypatch):
    from src import stage2_rollback
    cfg, grid, risk, plans, conflicts, layout, x = scenario(tmp_path)
    x[layout.blocks[0].atd_gene] = 800
    x[layout.blocks[0].speed_genes] = 14
    calls = []
    detector = stage2_rollback.detect_conflicts
    def tracked(plans, cfg, **kwargs):
        calls.append(len(plans))
        return detector(plans, cfg, **kwargs)
    monkeypatch.setattr(stage2_rollback, "detect_conflicts", tracked)
    result = decode_stage2_solution(x, plans, plans, layout, np.full(len(conflicts), strategy), cfg, grid, risk)
    final = detect_conflicts(result, cfg)
    assert calls == [len(plans), len(plans)]
    assert (count_conflict_pairs(final), len(final)) <= (count_conflict_pairs(conflicts), len(conflicts))
    assert plans[0].atd == 1000 and plans[0].speed_profile == [10]*5


def test_stage1_ignores_switch(tmp_path):
    cfg, grid, risk, plans, conflicts, _, _ = scenario(tmp_path)
    layout = build_stage1_decision_layout(plans, [0], conflicts, cfg, grid)
    x = (layout.lower+layout.upper)/2
    x[layout.blocks[0].strategy_gene] = 0
    x[layout.blocks[0].atd_gene] = 800
    enabled = decode_stage1_solution(x, plans, layout, cfg, grid, risk)
    cfg["optimization"]["stage2_global_rollback"] = False
    disabled = decode_stage1_solution(x, plans, layout, cfg, grid, risk)
    assert enabled[0] == disabled[0]
    assert not list(tmp_path.glob("actions_*.csv"))


def test_infeasible_route_keeps_infinite_fitness(tmp_path):
    cfg, grid, risk, plans, conflicts, layout, x = scenario(tmp_path)
    grid.obstacles[4, 5, 2] = True
    obj = PaperPopulationObjective(plans, plans, layout, cfg, grid, risk,
                                   PaperReference.from_initial(plans, risk, conflicts), 200, 2)
    assert obj.context_fitness(x, 7, np.full(len(conflicts), 2)) == float("inf")
    assert plans[0].path == plans[1].path


def test_rejected_speed_restores_previously_accepted_schedule(tmp_path, monkeypatch):
    from src import stage2_rollback
    cfg, grid, risk, plans, conflicts, layout, x = scenario(tmp_path)
    x[layout.blocks[0].atd_gene] = 800
    x[layout.blocks[0].speed_genes] = 14
    sampled = np.zeros(len(conflicts), dtype=int)
    sampled[-1] = 1
    # Force the coordination diagnostic: schedule improves, speed creates a new pair.
    extra = copy.copy(conflicts[0])
    extra.plan_b = 999
    detections = iter([conflicts, conflicts[:1], conflicts[:1], [conflicts[0], extra]])
    monkeypatch.setattr(stage2_rollback, "detect_conflicts", lambda *a, **kw: next(detections))
    result = decode_stage2_solution(x, plans, plans, layout, sampled, cfg, grid, risk)
    assert result[0].atd == 800
    assert result[0].speed_profile == [10]*5
    assert result[0].paper_stage2_strategies == (0,)
    logs = pd.read_csv(next(tmp_path.glob("actions_*.csv")))
    assert logs.accepted.tolist() == [True, False]
    assert logs.rollback_reason.iloc[-1] == "pairs_increased"
