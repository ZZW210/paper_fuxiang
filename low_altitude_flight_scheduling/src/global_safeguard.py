"""A_plus_global_safeguard: project enhancement, not disclosed paper logic."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from .adm_matching import (ADMResult, initialize_probability_matrix,
    sample_strategy_species, update_probability_matrix)
from .conflict_detection import count_conflict_pairs
from .fata import good_point_set, fata_vectorized_update
from .paper_optimization import (PaperPopulationObjective, InfeasiblePaperRoute,
    paper_fitness, stage2_flight_strategies)


@dataclass
class CachedEvaluation:
    fitness: float
    components: dict
    conflicts: list


def conflict_key(evaluation):
    if not np.isfinite(evaluation.fitness):
        return (float('inf'), float('inf'), float('inf'))
    return (count_conflict_pairs(evaluation.conflicts), evaluation.components['Nc'], evaluation.fitness)


def global_conflict_non_deterioration_accept(parent_eval, trial_eval):
    return conflict_key(trial_eval) < conflict_key(parent_eval)


def retain_candidate(parent, parent_context, parent_eval, trial, trial_context, trial_eval):
    accepted = global_conflict_non_deterioration_accept(parent_eval, trial_eval)
    return (trial.copy(), trial_context.copy(), trial_eval, True) if accepted else (
        parent.copy(), parent_context.copy(), parent_eval, False)


_OBJECTIVE = None


def _initialize(objective):
    global _OBJECTIVE
    _OBJECTIVE = objective


def _evaluate(task):
    position, generation, context = task
    try:
        evaluation = _OBJECTIVE.evaluation(position, generation, context)
        return CachedEvaluation(evaluation.fitness, evaluation.components, evaluation.conflicts)
    except InfeasiblePaperRoute:
        return CachedEvaluation(float('inf'), {}, [])


def fata_optimize_paper_with_safeguard(objective, layout, cfg, seed, remaining, callback=None):
    """Original MLF/LPS proposals; only parent-to-trial acceptance is added.

    Cached geometry/components are generation invariant. Fitness is repriced
    under current delta for parents, incumbent and conflict-lexicographic elite.
    Rejected trial contexts never contribute to the unchanged ADM formula.
    """
    from run_stage2_ab import origins
    population = int(cfg['fata']['NP'])
    max_gen = int(cfg['fata']['Ngen_max_stage2'])
    n_jobs = int(cfg['optimization']['n_jobs'])
    fraction = cfg['adm']['dominant_fraction']
    if population<2 or max_gen<1 or not 1<=n_jobs<=8 or not 0<fraction<=1:
        raise ValueError('Invalid FATA/ADM budget')
    dominant_no = max(1,round(population*fraction))
    lower, upper = layout.lower, layout.upper
    rng = np.random.default_rng(seed)
    flight = good_point_set(population,layout.dim,lower,upper)
    probability = initialize_probability_matrix(len(remaining))
    probability_history = [probability.copy()]
    generation_history, convergence, diagnostics, masks = [], [], [], []
    parents = parent_contexts = evaluations = None
    best_pos, best_context, best_eval = flight[0].copy(), None, None
    lex_eval = lex_pos = lex_context = None
    best_generation = 0
    worst_integral, best_integral = 0., float('inf')

    def reprice(evaluation,generation):
        if np.isfinite(evaluation.fitness):
            evaluation.fitness = paper_fitness(evaluation.components,objective.reference,cfg,generation,max_gen)

    executor = ProcessPoolExecutor(max_workers=n_jobs,initializer=_initialize,initargs=(objective,)) if n_jobs>1 else None
    try:
        if executor is None:
            _initialize(objective)
        for generation in range(1,max_gen+1):
            flight = np.clip(flight,lower,upper)
            trial_contexts = sample_strategy_species(probability,population,rng)
            tasks = [(flight[i],generation,trial_contexts[i]) for i in range(population)]
            trials = list(executor.map(_evaluate,tasks)) if executor else list(map(_evaluate,tasks))
            accepted = np.ones(population,dtype=bool) if generation==1 else np.zeros(population,dtype=bool)
            reasons = dict(rollback_due_to_pair_increase=0,rollback_due_to_point_increase=0,
                rollback_due_to_fitness_worse=0,new_pair_trials_rejected=0,
                rejected_lower_risk_equal_conflicts=0,rollback_due_to_infeasible_route=0)
            if generation==1:
                contexts = trial_contexts.copy()
                evaluations = trials
            else:
                contexts = parent_contexts.copy()
                for i in range(population):
                    reprice(evaluations[i],generation)
                    parent_key, trial_key = conflict_key(evaluations[i]),conflict_key(trials[i])
                    position, context, evaluation, accepted[i] = retain_candidate(
                        parents[i],parent_contexts[i],evaluations[i],flight[i],trial_contexts[i],trials[i])
                    if not accepted[i]:
                        if not np.isfinite(trials[i].fitness):
                            reasons['rollback_due_to_infeasible_route']+=1
                        elif trial_key[0]>parent_key[0]:
                            reasons['rollback_due_to_pair_increase']+=1
                        elif trial_key[1]>parent_key[1]:
                            reasons['rollback_due_to_point_increase']+=1
                        else:
                            reasons['rollback_due_to_fitness_worse']+=1
                        def pairs(e):
                            return {tuple(sorted((c.plan_a,c.plan_b))) for c in e.conflicts}
                        reasons['new_pair_trials_rejected']+=int(bool(pairs(trials[i])-pairs(evaluations[i])))
                        if trial_key[:2]==parent_key[:2] and np.isfinite(trials[i].fitness) and np.isfinite(evaluations[i].fitness):
                            reasons['rejected_lower_risk_equal_conflicts']+=int(
                                trials[i].components['ORISK']<evaluations[i].components['ORISK'])
                    flight[i],contexts[i],evaluations[i] = position,context,evaluation
            fitness = np.array([e.fitness for e in evaluations])
            if best_eval is not None:
                reprice(best_eval,generation)
            if lex_eval is not None:
                reprice(lex_eval,generation)
            for i,evaluation in enumerate(evaluations):
                if np.isfinite(evaluation.fitness) and (best_eval is None or evaluation.fitness<best_eval.fitness):
                    best_eval = CachedEvaluation(evaluation.fitness,evaluation.components,evaluation.conflicts)
                    best_pos,best_context,best_generation = flight[i].copy(),contexts[i].copy(),generation
                if lex_eval is None or conflict_key(evaluation)<conflict_key(lex_eval):
                    lex_eval = CachedEvaluation(evaluation.fitness,evaluation.components,evaluation.conflicts)
                    lex_pos,lex_context = flight[i].copy(),contexts[i].copy()
            ranked = np.argsort(fitness,kind='stable')
            ranked = ranked[np.isfinite(fitness[ranked]) & accepted[ranked]][:dominant_no]
            if len(ranked):
                probability = update_probability_matrix(probability,contexts[ranked],cfg['adm']['learning_rate'])
            probability_history.append(probability.copy())
            index = int(np.argmin(fitness))
            generation_history.append((generation,flight[index,layout.actor_genes].copy(),contexts[index].copy()))
            masks.append(accepted.copy())
            finite_keys = [conflict_key(e) for e in evaluations if np.isfinite(e.fitness)]
            best_score = best_eval.fitness if best_eval is not None else float('inf')
            convergence.append(best_score)
            row = dict(generation=generation,population_size=population,
                accepted_trials=int(accepted.sum()) if generation>1 else 0,
                rolled_back_trials=int((~accepted).sum()) if generation>1 else 0,
                acceptance_rate=float(accepted.mean()) if generation>1 else float('nan'),
                best_conflict_pairs=conflict_key(best_eval)[0] if best_eval else float('inf'),
                best_conflict_points=conflict_key(best_eval)[1] if best_eval else float('inf'),best_fitness=best_score,
                mean_conflict_pairs=float(np.mean([k[0] for k in finite_keys])) if finite_keys else float('inf'),
                mean_conflict_points=float(np.mean([k[1] for k in finite_keys])) if finite_keys else float('inf'),
                best_conflict_lexicographic_pairs=conflict_key(lex_eval)[0],
                best_conflict_lexicographic_points=conflict_key(lex_eval)[1],
                adm_learning_samples=len(ranked),**reasons)
            if best_eval:
                row.update(origins(remaining,best_eval.conflicts))
            diagnostics.append(row)
            if callback:
                callback(row)
            if generation==max_gen:
                break
            parents,parent_contexts = flight.copy(),contexts.copy()
            finite = fitness[np.isfinite(fitness)]
            if not len(finite):
                for i in range(population):
                    flight[i]=lower+rng.random()*(upper-lower)
                continue
            quality = np.where(np.isfinite(fitness),fitness,max(finite)+max(1.,abs(max(finite))))
            order = np.sort(quality)
            integral = float(np.trapezoid(order))
            worst_integral,best_integral = max(worst_integral,integral),min(best_integral,integral)
            eps = np.finfo(float).eps
            ip = (integral-worst_integral)/(best_integral-worst_integral+eps)
            a = np.tan(1.-generation/max_gen)
            b,worst = 1./a,float(order[-1])
            for i in range(population):
                para1 = a*rng.random(layout.dim)-a*rng.random(layout.dim)
                para2 = b*rng.random(layout.dim)-b*rng.random(layout.dim)
                p = (quality[i]-worst)/(best_score-worst+eps)
                if rng.random()>ip:
                    flight[i]=(upper-lower)*rng.random()+lower
                else:
                    fata_vectorized_update(flight,i,best_pos,para1,para2,p,rng,lower,upper,
                        cfg['fata']['Parf'],cfg.get('paper_performance',{}).get('fata_vectorized_update',False))
    finally:
        if executor:
            executor.shutdown(wait=True)
    if best_eval is None:
        raise RuntimeError('A+ found no feasible candidate')
    final = objective.evaluation(best_pos,max_gen,best_context)
    result = ADMResult(best_context,best_pos,final.plans,final.fitness,probability_history,convergence,
        stage2_flight_strategies(layout,best_context,best_pos[layout.actor_genes]),probability,
        fitness_evaluations=population*max_gen,best_generation=best_generation,generation_history=generation_history)
    return result,diagnostics,np.asarray(masks),dict(position=lex_pos,context=lex_context,
        key=conflict_key(lex_eval),output_policy='Unchanged paper-fitness incumbent among retained individuals; lex elite is diagnostic only')


def adm_fata_optimize_with_safeguard(stage1_plans,initial_plans,conflicts,layout,cfg,grid,risk_map,reference,seed=2025,callback=None):
    objective = PaperPopulationObjective(stage1_plans,initial_plans,layout,cfg,grid,risk_map,reference,
        cfg['fata']['Ngen_max_stage2'],2)
    return fata_optimize_paper_with_safeguard(objective,layout,cfg,seed,conflicts,callback)
