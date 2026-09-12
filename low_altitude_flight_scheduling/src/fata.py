from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

import numpy as np


Objective = Callable[[np.ndarray], float]
ObjectiveWithIter = Callable[[np.ndarray, int], float]


@dataclass
class FATAResult:
    best_position: np.ndarray
    best_fitness: float
    convergence: list[float]
    remaining_conflicts: list[float]
    delay_count: list[float]


@dataclass
class PaperFATAResult(FATAResult):
    best_context: np.ndarray | None


_PAPER_WORKER_OBJECTIVE = None
_PAPER_WORKER_CONTEXT_OBJECTIVE = None


def _initialize_paper_worker(objective, context_objective):
    global _PAPER_WORKER_OBJECTIVE, _PAPER_WORKER_CONTEXT_OBJECTIVE
    _PAPER_WORKER_OBJECTIVE = objective
    _PAPER_WORKER_CONTEXT_OBJECTIVE = context_objective


def _paper_worker_evaluate(task):
    vector, generation, context = task
    if _PAPER_WORKER_CONTEXT_OBJECTIVE is not None:
        return float(_PAPER_WORKER_CONTEXT_OBJECTIVE(vector, generation, context))
    return float(_PAPER_WORKER_OBJECTIVE(vector, generation))


def fata_optimize_paper(
    objective_with_iter,
    lb,
    ub,
    dim: int,
    population: int = 50,
    max_iter: int = 200,
    seed: int = 2025,
    parf: float = 0.2,
    n_jobs: int = 8,
    callback=None,
    generation_context=None,
    objective_with_context=None,
    on_generation_evaluated=None,
) -> PaperFATAResult:
    """FATA.m MLF/LPS, with Eq.(48) initialization and a generation-aware objective.

    Context hooks implement ADM without changing continuous position updates.
    Only objective evaluation runs in workers; RNG and selection stay ordered.
    """
    if dim < 1 or population < 2 or max_iter < 1 or n_jobs < 1:
        raise ValueError("Positive dimension, generations/jobs and population >= 2 required")
    lower = np.broadcast_to(np.asarray(lb, dtype=float), (dim,)).copy()
    upper = np.broadcast_to(np.asarray(ub, dtype=float), (dim,)).copy()
    if np.any(lower > upper):
        raise ValueError("Lower bounds exceed upper bounds")
    rng = np.random.default_rng(seed)
    flight = good_point_set(population, dim, lower, upper)
    best_pos = flight[0].copy()
    best_context = None
    best_score = float("inf")
    worst_integral, best_integral = 0.0, float("inf")
    convergence, remaining, delays = [], [], []
    executor = None
    if n_jobs > 1:
        executor = ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_initialize_paper_worker,
            initargs=(objective_with_iter, objective_with_context),
        )

    def evaluate(tasks):
        if executor is not None:
            return list(executor.map(_paper_worker_evaluate, tasks))
        return [float(objective_with_context(x, gen, context))
                if objective_with_context is not None
                else float(objective_with_iter(x, gen)) for x, gen, context in tasks]

    try:
        for generation in range(1, max_iter + 1):
            flight = np.clip(flight, lower, upper)
            contexts = generation_context(generation, population, rng) if generation_context else [None] * population
            tasks = [(flight[i], generation, contexts[i]) for i in range(population)]
            # The incumbent must be compared under this generation's delta too.
            has_incumbent = np.isfinite(best_score)
            if has_incumbent:
                tasks.append((best_pos, generation, best_context))
            scores = np.asarray(evaluate(tasks), dtype=float)
            fitness = scores[:population]
            if has_incumbent:
                best_score = float(scores[-1])
            for i in range(population):
                if fitness[i] < best_score:
                    best_score = float(fitness[i])
                    best_pos = flight[i].copy()
                    best_context = None if contexts[i] is None else np.array(contexts[i], copy=True)
            if on_generation_evaluated:
                on_generation_evaluated(generation, flight.copy(), fitness.copy(), contexts)
            convergence.append(best_score)
            if callback and np.isfinite(best_score):
                info = callback(best_pos.copy(), best_score, generation, best_context)
                remaining.append(float(info.get("remaining_conflicts", np.nan)))
                delays.append(float(info.get("delay_count", np.nan)))

            if generation == max_iter:
                break  # FATA.m's final unevaluated update cannot change its result.
            finite = fitness[np.isfinite(fitness)]
            if not len(finite):
                for i in range(population):
                    flight[i] = lower + rng.random() * (upper - lower)
                continue
            # Invalid routes have infinite fitness, not an added objective penalty.
            # A finite sentinel is used only to keep the MLF integral well defined.
            quality = np.where(np.isfinite(fitness), fitness, max(finite) + max(1.0, abs(max(finite))))
            order = np.sort(quality)
            integral = float(np.trapezoid(order))
            worst_integral = max(worst_integral, integral)
            best_integral = min(best_integral, integral)
            eps = np.finfo(float).eps
            ip = (integral - worst_integral) / (best_integral - worst_integral + eps)
            a = np.tan(1.0 - generation / max_iter)
            b = 1.0 / a
            worst = float(order[-1])
            for i in range(population):
                para1 = a * rng.random(dim) - a * rng.random(dim)
                para2 = b * rng.random(dim) - b * rng.random(dim)
                p = (quality[i] - worst) / (best_score - worst + eps)
                if rng.random() > ip:
                    # FATA.m uses scalar rand here (the same fraction in all axes).
                    flight[i] = (upper - lower) * rng.random() + lower
                else:
                    for j in range(dim):
                        num = int(np.floor(rng.random() * population))
                        if rng.random() < p:
                            flight[i, j] = best_pos[j] + flight[i, j] * para1[j]
                        else:
                            flight[i, j] = flight[num, j] + para2[j] * flight[i, j]
                            flight[i, j] = 0.5 * (parf + 1.0) * (lower[j] + upper[j]) - parf * flight[i, j]
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return PaperFATAResult(best_pos, best_score, convergence, remaining, delays, best_context)


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(np.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def _prime_for_dim(dim: int) -> int:
    k = max(5, 2 * dim + 3)
    while not _is_prime(k):
        k += 1
    return k


def good_point_set(n: int, dim: int, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    k = _prime_for_dim(dim)
    out = np.zeros((n, dim), dtype=float)
    for i in range(1, n + 1):
        for j in range(1, dim + 1):
            lam = np.mod(2.0 * np.cos(2.0 * np.pi * j / k) * i, 1.0)
            out[i - 1, j - 1] = lb[j - 1] + lam * (ub[j - 1] - lb[j - 1])
    return out


def fata_optimize(
    objective: Objective,
    lb: np.ndarray | list[float] | float,
    ub: np.ndarray | list[float] | float,
    dim: int,
    population: int = 50,
    max_iter: int = 200,
    seed: int = 2025,
    improved: bool = True,
    parf: float = 0.2,
    callback: Callable[[np.ndarray, float, int], dict[str, float]] | None = None,
    objective_with_iter: ObjectiveWithIter | None = None,
    initial_positions: np.ndarray | list[np.ndarray] | None = None,
) -> FATAResult:
    rng = np.random.default_rng(seed)
    lb_arr = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.asarray(lb, dtype=float)
    ub_arr = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.asarray(ub, dtype=float)
    no_p = int(population)
    if improved:
        flight = good_point_set(no_p, dim, lb_arr, ub_arr)
        flight += rng.normal(0, 0.015, size=flight.shape) * (ub_arr - lb_arr)
    else:
        flight = rng.random((no_p, dim)) * (ub_arr - lb_arr) + lb_arr
    flight = np.clip(flight, lb_arr, ub_arr)
    if initial_positions is not None:
        warm = np.atleast_2d(np.asarray(initial_positions, dtype=float))
        if warm.shape[1] != dim:
            raise ValueError(f"initial_positions has dimension {warm.shape[1]}, expected {dim}")
        warm_count = min(no_p, warm.shape[0])
        flight[:warm_count] = np.clip(warm[:warm_count], lb_arr, ub_arr)

    fitness = np.full(no_p, np.inf)
    best_pos = np.zeros(dim)
    best_score = float("inf")
    worst_integral = 0.0
    best_integral = float("inf")
    convergence: list[float] = []
    remaining_conflicts: list[float] = []
    delay_count: list[float] = []

    def _evaluate(x: np.ndarray, iteration: int) -> float:
        if objective_with_iter is not None:
            return float(objective_with_iter(x, iteration))
        return float(objective(x))

    for it in range(1, max_iter + 1):
        for i in range(no_p):
            flight[i] = np.clip(flight[i], lb_arr, ub_arr)
            fitness[i] = _evaluate(flight[i], it)
            if fitness[i] < best_score:
                best_score = float(fitness[i])
                best_pos = flight[i].copy()

        order = np.sort(fitness)
        worst_fitness = float(order[-1])
        integral = float(np.trapezoid(order))
        worst_integral = max(worst_integral, integral)
        best_integral = min(best_integral, integral)
        ip = (integral - worst_integral) / (best_integral - worst_integral + np.finfo(float).eps)
        progress = it / max_iter
        tan_arg = max(1e-6, 1.0 - progress)
        a = np.tan(tan_arg)
        b = 1.0 / max(np.tan(tan_arg), 1e-6)

        for i in range(no_p):
            para1 = a * rng.random(dim) - a * rng.random(dim)
            para2 = b * rng.random(dim) - b * rng.random(dim)
            p = (float(fitness[i]) - worst_fitness) / (best_score - worst_fitness + np.finfo(float).eps)
            if rng.random() > ip:
                flight[i] = rng.random(dim) * (ub_arr - lb_arr) + lb_arr
            else:
                for j in range(dim):
                    num = int(rng.integers(0, no_p))
                    if rng.random() < p:
                        flight[i, j] = best_pos[j] + flight[i, j] * para1[j]
                    else:
                        flight[i, j] = flight[num, j] + para2[j] * flight[i, j]
                        flight[i, j] = 0.5 * (parf + 1.0) * (lb_arr[j] + ub_arr[j]) - parf * flight[i, j]
            flight[i] = np.clip(flight[i], lb_arr, ub_arr)

        if improved and dim > 0:
            scale = (ub_arr - lb_arr) * max(0.015, 0.18 * (1.0 - progress))
            local_count = min(4, max(1, no_p // 12))
            for _ in range(local_count):
                candidate = np.clip(best_pos + rng.normal(0.0, scale, dim), lb_arr, ub_arr)
                candidate_fit = _evaluate(candidate, it)
                if candidate_fit < best_score:
                    best_score = float(candidate_fit)
                    best_pos = candidate.copy()
                    worst_idx = int(np.argmax(fitness))
                    flight[worst_idx] = candidate
                    fitness[worst_idx] = candidate_fit

        convergence.append(best_score)
        if callback:
            info = callback(best_pos, best_score, it)
            remaining_conflicts.append(float(info.get("remaining_conflicts", np.nan)))
            delay_count.append(float(info.get("delay_count", np.nan)))

    return FATAResult(best_pos, best_score, convergence, remaining_conflicts, delay_count)


def sphere_objective(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=float) ** 2))
