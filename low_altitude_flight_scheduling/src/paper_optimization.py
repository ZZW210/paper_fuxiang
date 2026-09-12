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
    norm = normalize_paper_objectives(components, reference)
    weights = cfg["fata"]
    delta = conflict_weight_delta(n_gen, n_gen_max, weights["gamma"])
    wc = weights["omega_c"]
    f_obj = (1.0 - wc) * (
        weights["omega_d"] * norm["Tdelay_norm"]
        + weights["omega_t"] * norm["Tair_norm"]
        + weights["omega_r"] * norm["ORISK_norm"]
    ) + wc * delta * norm["Nc_norm"] * norm["Tair_norm"]
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
    strategy_gene: int | None
    atd_gene: int
    speed_genes: slice
    segment_indices: list[int]
    reroute_genes: list[slice]
    reroute_intervals: list[tuple[int, int]]


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


def _build_layout(plans, flight_ids, conflicts, cfg, grid, stage):
    by_id = {p.id: p for p in plans}
    lower, upper, blocks = [], [], []
    window = max(1, int(cfg.get("paper_encoding", {}).get("local_window_segments", 4)))
    for fid in flight_ids:
        plan = by_id[fid]
        indices = []
        for conflict in conflicts:
            if fid not in (conflict.plan_a, conflict.plan_b):
                continue
            # Stage-2 paths can differ from the original route. Nearest-cell
            # projection is an encoding assumption, not a conflict-type rule.
            idx = min(range(len(plan.path)), key=lambda i: (sum((plan.path[i][j] - conflict.cell[j]) ** 2 for j in range(3)), i))
            indices.append(idx)
        strategy_gene = len(lower) if stage == 1 else None
        if stage == 1:
            lower.append(0.0)
            upper.append(np.nextafter(3.0, 0.0))
        atd_gene = len(lower)
        lo, hi = paper_atd_bounds(plan, cfg)
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
        for _ in intervals:
            route_genes.append(slice(len(lower), len(lower) + 3))
            # Paper x/y/z are 1-based [1,60]/[1,60]/[1,4]. Internal
            # coordinates subtract one: [0,59]/[0,59]/[0,3].
            lower.extend([0.0, 0.0, 0.0])
            upper.extend([float(s - 1) for s in grid.shape])
        blocks.append(PaperFlightBlock(fid, strategy_gene, atd_gene, speed_genes, segments, route_genes, intervals))
    return PaperDecisionLayout(blocks, np.asarray(lower), np.asarray(upper), len(conflicts))


def build_stage1_decision_layout(plans, key_ids, conflicts, cfg, grid):
    return _build_layout(plans, key_ids, conflicts, cfg, grid, 1)


def build_stage2_decision_layout(initial_plans, conflicts, cfg, grid):
    ids = sorted({fid for c in conflicts for fid in (c.plan_a, c.plan_b)})
    return _build_layout(initial_plans, ids, conflicts, cfg, grid, 2)


class InfeasiblePaperRoute(ValueError):
    pass


def _valid_route(path, plan, grid):
    return bool(path) and path[0] == plan.start and path[-1] == plan.goal and all(grid.is_free(c) for c in path) and all(
        max(abs(a[j] - b[j]) for j in range(3)) <= 1 and a != b for a, b in zip(path, path[1:])
    )


def _via_route(plan, block, vector, grid, risk_map, cfg, route_cache):
    path, speeds, cursor = [], [], 0
    for interval, genes in zip(block.reroute_intervals, block.reroute_genes):
        start, end = interval
        via = tuple(int(v) for v in np.rint(vector[genes]))
        if not grid.is_free(via):
            raise InfeasiblePaperRoute("Via-cell intersects an obstacle")
        endpoints = plan.path[start], plan.path[end]
        key = (endpoints, via)
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
        # Rerouting does not introduce a speed decision: inserted segments inherit
        # their replaced segment's speed, preserving untouched prefix/suffix.
        speeds.extend([plan.speed_profile[start]] * (len(local) - 1))
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
    for bi, block in enumerate(layout.blocks):
        strategy = int(np.floor(vector[block.strategy_gene])) if strategies is None else int(strategies[bi])
        if strategy == -1:
            continue
        if strategy not in (0, 1, 2):
            raise ValueError("Unknown paper strategy")
        base = original[block.flight_id]
        plan = copy.copy(base)
        plan.path = list(base.path)
        plan.speed_profile = list(base.speed_profile)
        plan.scheduled_etd = base.etd
        plan.paper_strategy = strategy
        plan.delay = 0.0
        plan.rerouted = False
        if strategy == 0:
            plan.atd = float(vector[block.atd_gene])
            plan.delay = plan.atd - base.etd
        elif strategy == 1:
            for segment, value in zip(block.segment_indices, vector[block.speed_genes]):
                plan.speed_profile[segment] = float(value)
        else:
            plan.path, plan.speed_profile = _via_route(base, block, vector, grid, risk_map, cfg, cache)
            plan.rerouted = plan.path != base.path
        if not _valid_route(plan.path, plan, grid):
            raise InfeasiblePaperRoute("Route violates endpoints, obstacles or 26-neighborhood")
        _recompute_paper_timing(plan, grid, risk_map)
        plan.changed = plan.rerouted or abs(plan.delay) > 1e-6 or plan.speed_profile != base.speed_profile
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
        strategies = context[self.layout.conflict_count:]
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
