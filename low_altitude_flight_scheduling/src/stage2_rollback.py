"""Experimental local-action protection; not a mechanism attributed to the paper."""
from __future__ import annotations

import copy
import csv
import os
from pathlib import Path

import numpy as np

from .conflict_detection import count_conflict_pairs, detect_conflicts
from .paper_optimization import (CandidatePlanView, InfeasiblePaperRoute, RerouteLRU,
                                 _recompute_paper_timing, _valid_route, _via_route,
                                 flight_strategies, flight_strategy_requirements)

COLUMNS = ["generation", "evaluation_id", "conflict_index", "actor_flight", "strategy",
           "pairs_before", "pairs_after", "points_before", "points_after", "accepted",
           "rollback_reason"]
_EVALUATION_ID = 0


def acceptance(before, after):
    if after < before:
        return True, ""
    return False, "pairs_increased" if after[0] > before[0] else "same_pairs_points_not_decreased"


def decode_with_rollback(vector, base_plans, initial_plans, layout, cfg, grid, risk_map,
                         strategies, route_cache=None, generation=0):
    global _EVALUATION_ID
    _EVALUATION_ID += 1
    vector = np.clip(np.asarray(vector, dtype=float), layout.lower, layout.upper)
    if vector.shape != (layout.dim,):
        raise ValueError("Incorrect paper decision vector dimension")
    original = {p.id: p for p in initial_plans}
    out = CandidatePlanView(base_plans)
    by_id = {p.id: p for p in base_plans}
    cache = RerouteLRU() if route_cache is None else route_cache
    cache.bind(grid, risk_map, cfg)
    requirements = flight_strategy_requirements(layout, strategies, vector[layout.actor_genes])
    rows = []
    try:
        for block in layout.blocks:
            required = requirements.get(block.flight_id, {})
            accepted_strategies = []
            for strategy in sorted(required):
                before_plan = out.overrides.get(block.flight_id, by_id[block.flight_id])
                before_conflicts = detect_conflicts(out, cfg, uncertain=True)
                before = (count_conflict_pairs(before_conflicts), len(before_conflicts))
                plan = copy.copy(before_plan)
                plan.path = list(before_plan.path)
                plan.speed_profile = list(before_plan.speed_profile)
                initial = original[plan.id]
                plan.scheduled_etd = initial.etd
                if strategy == 0:
                    plan.atd = float(vector[block.atd_gene])
                elif strategy == 1:
                    segments = {j for ci in required[1] for j in block.conflict_segments[ci]}
                    for segment, value in zip(block.segment_indices, vector[block.speed_genes]):
                        if segment in segments:
                            plan.speed_profile[segment] = float(value)
                else:
                    windows = {block.conflict_route_windows[ci] for ci in required[2]}
                    plan.path, plan.speed_profile = _via_route(
                        plan, block, vector, grid, risk_map, cfg, cache, windows)
                if not _valid_route(plan.path, plan, grid):
                    raise InfeasiblePaperRoute("Route violates endpoints, obstacles or 26-neighborhood")
                plan.delay = plan.atd - initial.etd
                plan.rerouted = plan.path != initial.path
                _recompute_paper_timing(plan, grid, risk_map)
                plan.changed = (plan.rerouted or abs(plan.delay) > 1e-6 or
                                len(plan.speed_profile) != len(initial.speed_profile) or
                                not np.allclose(plan.speed_profile, initial.speed_profile, atol=1e-6, rtol=0.0))
                out.overrides[plan.id] = plan
                after_conflicts = detect_conflicts(out, cfg, uncertain=True)
                after = (count_conflict_pairs(after_conflicts), len(after_conflicts))
                accepted, reason = acceptance(before, after)
                rows.append([generation, f"{os.getpid()}:{_EVALUATION_ID}",
                             ";".join(map(str, required[strategy])), plan.id,
                             ("schedule", "speed", "reroute")[strategy], before[0], after[0],
                             before[1], after[1], accepted, reason])
                if accepted:
                    accepted_strategies.append(strategy)
                    plan.paper_stage2_strategies = tuple(accepted_strategies)
                    plan.paper_strategies = tuple(sorted(set(flight_strategies(by_id[plan.id])) | set(accepted_strategies)))
                    plan.paper_strategy = plan.paper_strategies[0] if len(plan.paper_strategies) == 1 else None
                else:
                    # Only this actor was copied; restoring its reference restores the complete schedule.
                    out.overrides[plan.id] = before_plan
    finally:
        log_root = cfg.get("optimization", {}).get("stage2_rollback_log_dir")
        if log_root and rows:
            path = Path(log_root) / f"actions_{os.getpid()}.csv"
            exists = path.exists()
            with path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                if not exists:
                    writer.writerow(COLUMNS)
                writer.writerows(rows)
    return out
