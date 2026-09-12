"""Paper scheduling model, deliberately independent of engineering repair APIs."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from .astar_3d import astar_path
from .conflict_detection import Conflict, count_conflict_points, detect_conflicts
from .flight_plan import FlightPlan
from .grid import AirspaceGrid


def conflict_weight_delta(current_gen, max_gen, gamma=5.0):
    if max_gen <= 0 or not 0 <= current_gen <= max_gen:
        raise ValueError("Generation must lie in [0, max_gen]")
    return 1.0 - 0.1 * (current_gen / max_gen) ** gamma


@dataclass(frozen=True)
class PaperReference:
    delay_scale: float
    air_time_scale: float
    risk_scale: float
    conflict_scale: float

    @classmethod
    def from_initial(cls, plans, risk_map, conflicts, delay_max=1800.0):
        return cls(
            max(1.0, len(plans) * delay_max),
            max(np.finfo(float).eps, sum(p.eta_times[-1] - p.etd for p in plans)),
            max(np.finfo(float).eps, sum(float(risk_map[c]) for p in plans for c in p.path)),
            max(1, count_conflict_points(conflicts)),
        )


def paper_objective_components(plans, initial_plans, risk_map, conflicts, cfg):
    original = {p.id: p for p in initial_plans}
    air_times = [p.eta_times[-1] - p.atd for p in plans]
    return {
        "Tdelay": sum(abs(original[p.id].etd - p.atd) for p in plans),
        "Tair": sum(air_times),
        "ORISK": sum(float(risk_map[c]) for p in plans for c in p.path),
        "Nc": count_conflict_points(conflicts),
        "n_delay": sum(p.atd > original[p.id].etd + 1e-6 for p in plans),
        "n_battery": sum(t > cfg["optimization"]["t_battery"] for t in air_times),
    }


def normalize_paper_objectives(components, reference: PaperReference):
    return {
        "Tdelay_norm": components["Tdelay"] / reference.delay_scale,
        "Tair_norm": components["Tair"] / reference.air_time_scale,
        "ORISK_norm": components["ORISK"] / reference.risk_scale,
        "Nc_norm": components["Nc"] / reference.conflict_scale,
    }


def paper_fitness(components, reference, cfg, n_gen, n_gen_max):
    mode = cfg["optimization"].get("paper_objective_scale_mode", "raw_equation")
    if mode == "raw_equation":
        delay, air_time, risk, nc = (components[k] for k in ("Tdelay", "Tair", "ORISK", "Nc"))
    elif mode == "initial_reference_experimental":
        norm = normalize_paper_objectives(components, reference)
        delay, air_time, risk, nc = (norm[k] for k in ("Tdelay_norm", "Tair_norm", "ORISK_norm", "Nc_norm"))
    else:
        raise ValueError(f"Unknown paper objective scale mode: {mode}")
    weights = cfg["fata"]
    delta = conflict_weight_delta(n_gen, n_gen_max, weights["gamma"])
    wc = weights["omega_c"]
    f_obj = (1.0 - wc) * (
        weights["omega_d"] * delay
        + weights["omega_t"] * air_time
        + weights["omega_r"] * risk
    ) + wc * delta * nc * air_time
    return float(f_obj + 1000 * components["n_battery"] + 100 * components["n_delay"])


def paper_atd_bounds(plan, cfg):
    opt = cfg["optimization"]
    lo, hi = opt["t_ATD_range"]
    lower = max(lo, plan.etd - opt["t_delay_max"])
    upper = min(hi, plan.etd + opt["t_delay_max"])
    if lower > upper:
        raise ValueError(f"Flight {plan.id} has no feasible ATD interval")
    return float(lower), float(upper)


@dataclass
class PaperFlightBlock:
    flight_id: int
    activation_genes: slice | None
    atd_gene: int
    speed_genes: slice
    segment_indices: list[int]
    reroute_genes: list[slice]
    reroute_intervals: list[tuple[int, int]]
    incident_conflicts: list[int]
    conflict_segments: dict[int, list[int]]
    conflict_route_windows: dict[int, int]
    reroute_enable_genes: list[int]


def decode_activation_genes(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,):
        raise ValueError("Three activation genes required")
    active = values >= 0.5
    if not active.any():
        active[int(np.argmax(values))] = True
    return tuple(int(s) for s in np.flatnonzero(active))


def flight_strategies(plan):
    recorded = getattr(plan, "paper_strategies", None)
    if recorded is not None:
        return tuple(recorded)
    return () if plan.paper_strategy is None else (int(plan.paper_strategy),)


def stage2_flight_strategies(layout, sampled_species):
    species = np.asarray(sampled_species, dtype=int)
    if species.shape != (layout.conflict_count,) or np.any((species < 0) | (species > 2)):
        raise ValueError("One sampled strategy required per remaining conflict point")
    return {b.flight_id: tuple(sorted({int(species[i]) for i in b.incident_conflicts})) for b in layout.blocks}


@dataclass
class PaperDecisionLayout:
    blocks: list[PaperFlightBlock]
    lower: np.ndarray
    upper: np.ndarray
    conflict_count: int = 0

    @property
    def dim(self):
        return len(self.lower)


def _local_intervals(indices, path_length, window):
    intervals = sorted((max(0, i - window), min(path_length - 1, i + window)) for i in indices)
    merged = []
    for start, end in intervals:
        if start == end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _build_layout(plans, flight_ids, conflicts, cfg, grid, stage, initial_plans=None):
    by_id = {p.id: p for p in plans}
    original = {p.id: p for p in (plans if initial_plans is None else initial_plans)}
    lower, upper, blocks = [], [], []
    window = max(1, int(cfg.get("paper_encoding", {}).get("local_window_segments", 4)))
    for fid in flight_ids:
        plan = by_id[fid]
        indices, conflict_indices, conflict_segments = [], {}, {}
        for ci, conflict in enumerate(conflicts):
            if fid not in (conflict.plan_a, conflict.plan_b):
                continue
            idx = conflict.idx_a if fid == conflict.plan_a else conflict.idx_b
            if not 0 <= idx < len(plan.path) or plan.path[idx] != conflict.cell:
                raise ValueError("Conflict indices must refer to the stage input path")
            indices.append(idx)
            conflict_indices[ci] = idx
            conflict_segments[ci] = list(range(max(0, idx - window), min(len(plan.path) - 1, idx + window)))
        activation_genes = slice(len(lower), len(lower) + 3) if stage == 1 else None
        if stage == 1:
            lower.extend([0.0] * 3)
            upper.extend([1.0] * 3)
        atd_gene = len(lower)
        lo, hi = paper_atd_bounds(original[fid], cfg)
        lower.append(lo)
        upper.append(hi)
        if stage == 1:
            segments = list(range(len(plan.path) - 1))
        else:
            segments = sorted({j for i in indices for j in range(max(0, i - window), min(len(plan.path) - 1, i + window))})
        speed_genes = slice(len(lower), len(lower) + len(segments))
        lower.extend([cfg["optimization"]["speed_range"][0]] * len(segments))
        upper.extend([cfg["optimization"]["speed_range"][1]] * len(segments))
        intervals = _local_intervals(indices, len(plan.path), window)
        if stage == 1 and not intervals and len(plan.path) > 1:
            intervals = _local_intervals([len(plan.path) // 2], len(plan.path), window)
        route_genes = []
        enable_genes = []
        for _ in intervals:
            if stage == 2:
                enable_genes.append(len(lower))
                lower.append(0.0)
                upper.append(1.0)
            route_genes.append(slice(len(lower), len(lower) + 3))
            # Paper x/y/z are 1-based [1,60]/[1,60]/[1,4]. Internal
            # coordinates subtract one: [0,59]/[0,59]/[0,3].
            lower.extend([0.0, 0.0, 0.0])
            upper.extend([float(s - 1) for s in grid.shape])
        route_windows = {ci: next(wi for wi, (start, end) in enumerate(intervals) if start <= idx <= end)
                         for ci, idx in conflict_indices.items() if intervals}
        blocks.append(PaperFlightBlock(fid, activation_genes, atd_gene, speed_genes, segments,
            route_genes, intervals, list(conflict_indices), conflict_segments, route_windows, enable_genes))
    return PaperDecisionLayout(blocks, np.asarray(lower), np.asarray(upper), len(conflicts))


def build_stage1_decision_layout(plans, key_ids, conflicts, cfg, grid):
    return _build_layout(plans, key_ids, conflicts, cfg, grid, 1)


def build_stage2_decision_layout(stage1_plans, conflicts, cfg, grid, initial_plans=None):
    ids = sorted({fid for c in conflicts for fid in (c.plan_a, c.plan_b)})
    return _build_layout(stage1_plans, ids, conflicts, cfg, grid, 2, initial_plans)


class InfeasiblePaperRoute(ValueError):
    pass


def _valid_route(path, plan, grid):
    return bool(path) and path[0] == plan.start and path[-1] == plan.goal and all(grid.is_free(c) for c in path) and all(
        max(abs(a[j] - b[j]) for j in range(3)) <= 1 and a != b for a, b in zip(path, path[1:])
    )


def _via_route(plan, block, vector, grid, risk_map, cfg, route_cache, active_windows=None):
    path, speeds, cursor = [], [], 0
    for wi, (interval, genes) in enumerate(zip(block.reroute_intervals, block.reroute_genes)):
        if active_windows is not None and wi not in active_windows:
            continue
        start, end = interval
        via = tuple(int(v) for v in np.rint(vector[genes]))
        if not grid.is_free(via):
            raise InfeasiblePaperRoute("Via-cell intersects an obstacle")
        endpoints = plan.path[start], plan.path[end]
        original_local = plan.path[start:end + 1]
        if via == original_local[len(original_local) // 2]:
            local = original_local
        else:
            local = None
        key = (endpoints, via)
        if local is None:
            local = route_cache.get(key)
        if local is None:
            options = dict(alpha_r=cfg["risk"]["alpha_r"], alpha_l=cfg["risk"]["alpha_L"])
            first = astar_path(grid, endpoints[0], via, risk_map, **options)
            second = astar_path(grid, via, endpoints[1], risk_map, **options)
            if not first or not second:
                raise InfeasiblePaperRoute("Via-cell is unreachable")
            local = first + second[1:]
            if len(route_cache) >= 4096:
                route_cache.clear()
            route_cache[key] = local
        path.extend(plan.path[cursor:start])
        speeds.extend(plan.speed_profile[cursor:start])
        path.extend(local[:-1])
        if local == original_local:
            speeds.extend(plan.speed_profile[start:end])
        else:
            # Keep the current speed adjustment on replacement geometry by
            # normalized arc-length projection; this adds no speed decision.
            old_dist = np.linalg.norm(np.diff(np.asarray(original_local), axis=0) * grid.cell_size, axis=1)
            new_dist = np.linalg.norm(np.diff(np.asarray(local), axis=0) * grid.cell_size, axis=1)
            old_edges = np.concatenate([[0.0], np.cumsum(old_dist)]) / old_dist.sum()
            new_mid = (np.cumsum(new_dist) - 0.5 * new_dist) / new_dist.sum()
            projected = np.searchsorted(old_edges[1:], new_mid, side="right").clip(0, end - start - 1)
            speeds.extend([plan.speed_profile[start + int(i)] for i in projected])
        cursor = end
    path.extend(plan.path[cursor:])
    speeds.extend(plan.speed_profile[cursor:])
    return path, speeds


def _recompute_paper_timing(plan, grid, risk_map):
    distances = np.linalg.norm(np.diff(np.asarray(plan.path, dtype=float), axis=0) * np.asarray(grid.cell_size), axis=1)
    plan.eta_times = [plan.atd] + (plan.atd + np.cumsum(distances / np.asarray(plan.speed_profile))).tolist()
    plan.total_air_time = plan.eta_times[-1] - plan.atd
    plan.risk_sum = float(sum(risk_map[c] for c in plan.path))


def _decode(vector, base_plans, initial_plans, layout, cfg, grid, risk_map, strategies=None, route_cache=None):
    vector = np.clip(np.asarray(vector, dtype=float), layout.lower, layout.upper)
    if vector.shape != (layout.dim,):
        raise ValueError("Incorrect paper decision vector dimension")
    original = {p.id: p for p in initial_plans}
    out = list(base_plans)
    slots = {p.id: i for i, p in enumerate(out)}
    cache = {} if route_cache is None else route_cache
    matches = None if strategies is None else stage2_flight_strategies(layout, strategies)
    for block in layout.blocks:
        active = decode_activation_genes(vector[block.activation_genes]) if matches is None else matches[block.flight_id]
        base = base_plans[slots[block.flight_id]]
        initial = original[block.flight_id]
        plan = copy.copy(base)
        plan.path = list(base.path)
        plan.speed_profile = list(base.speed_profile)
        plan.scheduled_etd = initial.etd
        if 0 in active:
            plan.atd = float(vector[block.atd_gene])
        if 1 in active:
            selected_segments = set(block.segment_indices) if matches is None else {
                j for ci in block.incident_conflicts if strategies[ci] == 1 for j in block.conflict_segments[ci]}
            for segment, value in zip(block.segment_indices, vector[block.speed_genes]):
                if segment in selected_segments:
                    plan.speed_profile[segment] = float(value)
        if 2 in active:
            active_windows = None if matches is None else {
                block.conflict_route_windows[ci] for ci in block.incident_conflicts if strategies[ci] == 2
                and vector[block.reroute_enable_genes[block.conflict_route_windows[ci]]] >= 0.5}
            plan.path, plan.speed_profile = _via_route(plan, block, vector, grid, risk_map, cfg, cache, active_windows)
        plan.paper_stage2_strategies = active if matches is not None else ()
        plan.paper_strategies = tuple(sorted(set(flight_strategies(base)) | set(active)))
        plan.paper_strategy = plan.paper_strategies[0] if len(plan.paper_strategies) == 1 else None
        plan.delay = plan.atd - initial.etd
        plan.rerouted = plan.path != initial.path
        if not _valid_route(plan.path, plan, grid):
            raise InfeasiblePaperRoute("Route violates endpoints, obstacles or 26-neighborhood")
        _recompute_paper_timing(plan, grid, risk_map)
        plan.changed = plan.rerouted or abs(plan.delay) > 1e-6 or len(plan.speed_profile) != len(initial.speed_profile) or not np.allclose(plan.speed_profile, initial.speed_profile, atol=1e-6, rtol=0.0)
        out[slots[plan.id]] = plan
    return out


def decode_stage1_solution(vector, plans, layout, cfg, grid, risk_map, route_cache=None):
    return _decode(vector, plans, plans, layout, cfg, grid, risk_map, route_cache=route_cache)


def decode_stage2_solution(vector, stage1_plans, initial_plans, layout, strategies, cfg, grid, risk_map, route_cache=None):
    return _decode(vector, stage1_plans, initial_plans, layout, cfg, grid, risk_map, strategies, route_cache)


@dataclass
class PaperEvaluation:
    fitness: float
    components: dict
    plans: list[FlightPlan]
    conflicts: list[Conflict]
    delta: float


def _evaluate(plans, initial_plans, cfg, risk_map, reference, n_gen, n_gen_max):
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    components = paper_objective_components(plans, initial_plans, risk_map, conflicts, cfg)
    return PaperEvaluation(paper_fitness(components, reference, cfg, n_gen, n_gen_max), components, plans, conflicts,
                           conflict_weight_delta(n_gen, n_gen_max, cfg["fata"]["gamma"]))


def evaluate_stage1_paper_solution(vector, plans, layout, cfg, grid, risk_map, reference, n_gen, n_gen_max, route_cache=None):
    decoded = decode_stage1_solution(vector, plans, layout, cfg, grid, risk_map, route_cache)
    return _evaluate(decoded, plans, cfg, risk_map, reference, n_gen, n_gen_max)


def evaluate_stage2_paper_solution(vector, stage1_plans, initial_plans, layout, strategies, cfg, grid, risk_map, reference, n_gen, n_gen_max, route_cache=None):
    decoded = decode_stage2_solution(vector, stage1_plans, initial_plans, layout, strategies, cfg, grid, risk_map, route_cache)
    return _evaluate(decoded, initial_plans, cfg, risk_map, reference, n_gen, n_gen_max)


@dataclass
class PaperPopulationObjective:
    base_plans: list[FlightPlan]
    initial_plans: list[FlightPlan]
    layout: PaperDecisionLayout
    cfg: dict
    grid: AirspaceGrid
    risk_map: np.ndarray
    reference: PaperReference
    max_gen: int
    stage: int
    route_cache: dict = field(default_factory=dict)

    def evaluation(self, vector, generation, context=None):
        if self.stage == 1:
            return evaluate_stage1_paper_solution(vector, self.initial_plans, self.layout, self.cfg, self.grid,
                self.risk_map, self.reference, generation, self.max_gen, self.route_cache)
        strategies = context
        return evaluate_stage2_paper_solution(vector, self.base_plans, self.initial_plans, self.layout, strategies,
            self.cfg, self.grid, self.risk_map, self.reference, generation, self.max_gen, self.route_cache)

    def fitness(self, vector, generation):
        try:
            return self.evaluation(vector, generation).fitness
        except InfeasiblePaperRoute:
            return float("inf")

    def context_fitness(self, vector, generation, context):
        try:
            return self.evaluation(vector, generation, context).fitness
        except InfeasiblePaperRoute:
            return float("inf")
