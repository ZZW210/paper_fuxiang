"""Conflict-point ADM sampling coupled to the paper FATA population."""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np

from . import fata
from .conflict_detection import Conflict
from .paper_optimization import PaperPopulationObjective


class Strategy(IntEnum):
    SCHEDULING = 0
    SPEED = 1
    REROUTING = 2


@dataclass
class ConflictStrategyUnit:
    conflict_id: int
    conflict: Conflict
    probability: np.ndarray


@dataclass
class ADMSpecies:
    strategies: np.ndarray
    decision_vector: np.ndarray
    fitness: float


@dataclass
class ADMResult:
    best_strategies: np.ndarray
    best_decision_vector: np.ndarray
    best_plans: list
    best_fitness: float
    probability_history: list[np.ndarray]
    convergence: list[float]
    best_flight_strategies: dict[int, int]
    final_probability: np.ndarray
    conflict_owners: list[int]


def initialize_probability_matrix(n_conflicts):
    return np.full((n_conflicts, 3), 1.0 / 3.0)


def sample_strategy_species(probability, population, rng):
    if probability.ndim != 2 or probability.shape[1] != 3:
        raise ValueError("Probability matrix must have three strategy columns")
    draws = rng.random((population, len(probability)))
    return np.sum(draws[:, :, None] >= np.cumsum(probability, axis=1)[None, :, :], axis=2).clip(0, 2)


def enforce_single_strategy_per_flight(conflicts, sampled_strategies, probability, owners=None):
    """Reconcile sampled proposals by probability-weighted votes, with ID ties.

    Endpoint ownership and reconciliation are implementation assumptions. Voting
    only on sampled labels preserves exploration at the uniform initialization;
    an unconditional argmax of P alone would ignore every sampled species.
    """
    if owners is None:
        owners = [c.plan_a if i % 2 == 0 else c.plan_b for i, c in enumerate(conflicts)]
    if len(owners) != len(conflicts) or len(sampled_strategies) != len(conflicts):
        raise ValueError("Each conflict needs an owner and a strategy")
    scores = {}
    for i, (conflict, owner, strategy) in enumerate(zip(conflicts, owners, sampled_strategies)):
        if owner not in (conflict.plan_a, conflict.plan_b):
            raise ValueError("Owner must be an endpoint of the conflict")
        vote = scores.setdefault(owner, np.zeros(3))
        vote[int(strategy)] += probability[i, int(strategy)]
    flight_strategies = {fid: int(np.argmax(score)) for fid, score in scores.items()}
    consistent = np.asarray([flight_strategies[fid] for fid in owners], dtype=int)
    return consistent, flight_strategies


def update_probability_matrix(probability, dominant_species, learning_rate=0.5):
    dominant_species = np.asarray(dominant_species, dtype=int)
    if dominant_species.ndim != 2 or dominant_species.shape[1] != len(probability) or not len(dominant_species):
        raise ValueError("Dominant species must have shape (dominant_no, n_conflicts)")
    if not 0 <= learning_rate <= 1 or np.any((dominant_species < 0) | (dominant_species > 2)):
        raise ValueError("Invalid learning rate or strategy")
    frequencies = np.stack([(dominant_species == s).mean(axis=0) for s in range(3)], axis=1)
    updated = (1.0 - learning_rate) * probability + learning_rate * frequencies
    return updated / updated.sum(axis=1, keepdims=True)


def adm_fata_optimize(stage1_plans, initial_plans, conflicts, layout, cfg, grid, risk_map, reference,
                      seed=2025, callback=None):
    if not conflicts:
        raise ValueError("ADM requires remaining conflict points")
    population = int(cfg["fata"]["NP"])
    max_gen = int(cfg["fata"]["Ngen_max_stage2"])
    probability = initialize_probability_matrix(len(conflicts))
    history = [probability.copy()]
    owners = [c.plan_a if i % 2 == 0 else c.plan_b for i, c in enumerate(conflicts)]
    objective = PaperPopulationObjective(stage1_plans, initial_plans, layout, cfg, grid, risk_map,
                                          reference, max_gen, 2)
    fraction = float(cfg["adm"]["dominant_fraction"])
    if not 0 < fraction <= 1:
        raise ValueError("dominant_fraction must lie in (0,1]")
    dominant_no = max(1, round(population * fraction))

    def generate_context(generation, size, rng):
        sampled = sample_strategy_species(probability, size, rng)
        contexts = []
        for proposals in sampled:
            consistent, flight_strategies = enforce_single_strategy_per_flight(conflicts, proposals, probability, owners)
            contexts.append(np.concatenate([consistent, [flight_strategies.get(b.flight_id, -1) for b in layout.blocks]]))
        return np.asarray(contexts, dtype=int)

    def update(generation, positions, fitness, contexts):
        nonlocal probability
        # Infeasible individuals cannot teach ADM. This is route feasibility,
        # not a conflict-decrease acceptance rule.
        ranked = np.argsort(fitness, kind="stable")
        ranked = ranked[np.isfinite(fitness[ranked])][:dominant_no]
        if len(ranked):
            species = np.asarray(contexts)[ranked, :len(conflicts)]
            probability = update_probability_matrix(probability, species, cfg["adm"]["learning_rate"])
        history.append(probability.copy())

    result = fata.fata_optimize_paper(
        objective.fitness, layout.lower, layout.upper, layout.dim,
        population=population, max_iter=max_gen, seed=seed,
        parf=cfg["fata"]["Parf"], n_jobs=cfg["optimization"]["n_jobs"],
        objective_with_context=objective.context_fitness, generation_context=generate_context,
        on_generation_evaluated=update, callback=callback,
    )
    if not np.isfinite(result.best_fitness):
        raise RuntimeError("Stage 2 found no geometrically feasible candidate")
    evaluation = objective.evaluation(result.best_position, max_gen, result.best_context)
    flight_strategies = {block.flight_id: int(s) for block, s in zip(layout.blocks, result.best_context[len(conflicts):]) if s >= 0}
    return ADMResult(result.best_context[:len(conflicts)], result.best_position, evaluation.plans,
                     evaluation.fitness, history, result.convergence, flight_strategies, probability, owners)
