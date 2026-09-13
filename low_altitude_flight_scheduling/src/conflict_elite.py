"""GCEP is additional project search memory, not an original paper mechanism."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .fata import good_point_set, fata_vectorized_update, PaperFATAResult
from .paper_optimization import InfeasiblePaperRoute, paper_fitness
from .conflict_detection import count_conflict_pairs


@dataclass
class Evaluation:
    fitness: float
    components: dict
    conflicts: list
    changed_flights: int


def key(evaluation):
    if not np.isfinite(evaluation.fitness):
        return (float('inf'),)*3
    return count_conflict_pairs(evaluation.conflicts),evaluation.components['Nc'],evaluation.fitness


class ConflictEliteArchive:
    def __init__(self,objective):
        self.objective=objective
        self.position=self.context=self.evaluation=self.plans=None
        self.generation=0

    def reprice(self,generation):
        if self.evaluation is not None:
            self.evaluation.fitness=paper_fitness(self.evaluation.components,self.objective.reference,
                self.objective.cfg,generation,self.objective.max_gen)

    def consider(self,position,context,evaluation,generation):
        self.reprice(generation)
        if np.isfinite(evaluation.fitness) and (self.evaluation is None or key(evaluation)<key(self.evaluation)):
            self.position,self.context=position.copy(),context.copy()
            self.evaluation=deepcopy(evaluation)
            # Only a new archive entry is decoded, never a historical reprice.
            self.plans=deepcopy(list(self.objective.evaluation(position,generation,context).plans))
            self.generation=generation
            return True
        return False

    def present(self,positions,contexts):
        return self.position is not None and any(np.allclose(p,self.position,rtol=0.,atol=1e-12)
            and np.array_equal(c,self.context) for p,c in zip(positions,contexts))

    def inject(self,positions,contexts,evaluations):
        if self.evaluation is None or self.present(positions,contexts):
            return None
        index=max(range(len(evaluations)),key=lambda i:key(evaluations[i]))
        replaced=key(evaluations[index])
        positions[index],contexts[index]=self.position.copy(),self.context.copy()
        evaluations[index]=deepcopy(self.evaluation)
        return index,replaced


_OBJECTIVE=None


def _initialize(objective):
    global _OBJECTIVE
    _OBJECTIVE=objective


def _evaluate(task):
    from .paper_scheduler import _physically_changed
    position,generation,context=task
    try:
        e=_OBJECTIVE.evaluation(position,generation,context)
        changed=sum(_physically_changed(a,b) for a,b in zip(_OBJECTIVE.initial_plans,e.plans))
        return Evaluation(e.fitness,e.components,e.conflicts,changed)
    except InfeasiblePaperRoute:
        return Evaluation(float('inf'),{},[],0)


def run_diagnostic_fata(objective_with_iter,lb,ub,dim,population=50,max_iter=200,seed=2025,
    parf=.2,n_jobs=8,callback=None,generation_context=None,objective_with_context=None,
    on_generation_evaluated=None,vectorized_update=False,*,enable_elite=False,records=None):
    """Original proposal/ADM path; optional coherent evaluated-population injection.

    Injection follows normal evaluation, incumbent selection and ADM learning.
    It precedes the next MLF/LPS proposal, so injected copies are never frozen.
    No trial is rejected. Historical archive geometry is repriced, not reevaluated.
    """
    objective=objective_with_context.__self__
    lower=np.broadcast_to(np.asarray(lb,float),(dim,)).copy()
    upper=np.broadcast_to(np.asarray(ub,float),(dim,)).copy()
    rng=np.random.default_rng(seed)
    flight=good_point_set(population,dim,lower,upper)
    best_pos=flight[0].copy();best_context=best_eval=None;best_generation=0
    archive=ConflictEliteArchive(objective)
    worst_integral,best_integral=0.,float('inf')
    convergence,remaining,delays,history,stats=[],[],[],[],[]
    first_zero=None;reinjections=0
    executor=ProcessPoolExecutor(max_workers=n_jobs,initializer=_initialize,initargs=(objective,)) if n_jobs>1 else None
    def reprice(e,generation):
        if e is not None and np.isfinite(e.fitness):
            e.fitness=paper_fitness(e.components,objective.reference,objective.cfg,generation,max_iter)
    try:
        if executor is None:_initialize(objective)
        for generation in range(1,max_iter+1):
            flight=np.clip(flight,lower,upper)
            contexts=np.asarray(generation_context(generation,population,rng)).copy()
            tasks=[(flight[i],generation,contexts[i]) for i in range(population)]
            evaluations=list(executor.map(_evaluate,tasks)) if executor else list(map(_evaluate,tasks))
            reprice(best_eval,generation)
            fitness=np.asarray([e.fitness for e in evaluations])
            for i,e in enumerate(evaluations):
                if np.isfinite(e.fitness) and (best_eval is None or e.fitness<best_eval.fitness):
                    best_pos,best_context,best_eval=flight[i].copy(),contexts[i].copy(),deepcopy(e)
                    best_generation=generation
            # Exactly the reproduced ADM callback, before archive/injection.
            if on_generation_evaluated:
                on_generation_evaluated(generation,flight.copy(),fitness.copy(),contexts)
            finite_indices=[i for i,e in enumerate(evaluations) if np.isfinite(e.fitness)]
            zeros=[i for i in finite_indices if evaluations[i].components['Nc']==0]
            if zeros and first_zero is None:
                i=zeros[0];e=evaluations[i]
                first_zero=dict(generation=generation,fitness=e.fitness,components=deepcopy(e.components),
                    changed_flights=e.changed_flights,position=flight[i].copy(),context=contexts[i].copy(),
                    plans=deepcopy(list(objective.evaluation(flight[i],generation,contexts[i]).plans)))
            archive.reprice(generation)
            if finite_indices:
                i=min(finite_indices,key=lambda i:key(evaluations[i]))
                archive.consider(flight[i],contexts[i],evaluations[i],generation)
            present=archive.present(flight,contexts)
            current_keys=[key(evaluations[i]) for i in finite_indices]
            pre_points=[evaluations[i].components['Nc'] for i in finite_indices]
            injection=archive.inject(flight,contexts,evaluations) if enable_elite else None
            if injection is not None:reinjections+=1
            # Initialization / normal candidate metrics are pre-injection.
            best_conflict=min(current_keys) if current_keys else (float('inf'),)*3
            # Use original pre-injection evaluations for means/min/zero counts.
            stat=dict(generation=generation,best_paper_fitness=best_eval.fitness if best_eval else float('inf'),
                best_conflict_pairs=best_conflict[0],best_conflict_points=best_conflict[1],
                mean_conflict_pairs=float(np.mean([k[0] for k in current_keys])) if current_keys else float('inf'),
                mean_conflict_points=float(np.mean(pre_points)) if pre_points else float('inf'),
                min_conflict_points=min(pre_points) if pre_points else float('inf'),
                zero_conflict_individual_count=len(zeros),elite_present_before_injection=present,
                elite_reinjected=injection is not None,elite_replacement_count_cumulative=reinjections)
            stats.append(stat)
            if archive.evaluation is not None:
                e=archive.evaluation;comp=e.components
                replaced=injection[1] if injection else (None,None,None)
                history.append(dict(generation=generation,elite_pairs=key(e)[0],elite_points=comp['Nc'],
                    elite_paper_fitness=e.fitness,elite_risk_increase=comp['ORISK'],
                    elite_changed_flights=e.changed_flights,elite_delayed_flights=comp['n_delay'],
                    elite_generation_origin=archive.generation,elite_in_population=archive.present(flight,contexts),
                    elite_reinjected=injection is not None,replaced_individual_pairs=replaced[0],
                    replaced_individual_points=replaced[1],replaced_individual_fitness=replaced[2]))
            best_score=best_eval.fitness if best_eval else float('inf')
            convergence.append(best_score)
            if callback and np.isfinite(best_score):
                info=callback(best_pos.copy(),best_score,generation,best_context)
                remaining.append(float(info.get('remaining_conflicts',np.nan)))
                delays.append(float(info.get('delay_count',np.nan)))
            if generation%20==0:print(f'GCEP={enable_elite} gen={generation} best={stat["best_conflict_points"]} archive={key(archive.evaluation)[1] if archive.evaluation else "NA"}',flush=True)
            if generation==max_iter:break
            # Injection changes only the saved population; proposals remain MLF/LPS.
            fitness=np.asarray([e.fitness for e in evaluations])
            finite=fitness[np.isfinite(fitness)]
            if not len(finite):
                for i in range(population):flight[i]=lower+rng.random()*(upper-lower)
                continue
            quality=np.where(np.isfinite(fitness),fitness,max(finite)+max(1.,abs(max(finite))))
            order=np.sort(quality);integral=float(np.trapezoid(order))
            worst_integral=max(worst_integral,integral);best_integral=min(best_integral,integral)
            eps=np.finfo(float).eps;ip=(integral-worst_integral)/(best_integral-worst_integral+eps)
            a=np.tan(1.-generation/max_iter);b=1./a;worst=float(order[-1])
            for i in range(population):
                para1=a*rng.random(dim)-a*rng.random(dim);para2=b*rng.random(dim)-b*rng.random(dim)
                p=(quality[i]-worst)/(best_score-worst+eps)
                if rng.random()>ip:flight[i]=(upper-lower)*rng.random()+lower
                else:fata_vectorized_update(flight,i,best_pos,para1,para2,p,rng,lower,upper,parf,vectorized_update)
    finally:
        if executor:executor.shutdown(wait=True)
    if records is not None:
        records.update(archive=archive,stats=stats,history=history,first_zero=first_zero,
            reinjections=reinjections,ever_zero=first_zero is not None)
    return PaperFATAResult(best_pos,best_eval.fitness,convergence,remaining,delays,best_context,
        fitness_evaluations=population*max_iter,best_generation=best_generation)
