from types import SimpleNamespace

import numpy as np
import pytest

from src.global_safeguard import (CachedEvaluation, global_conflict_non_deterioration_accept,
    retain_candidate)
from src.fata import fata_optimize_paper


def evaluation(pairs,points,fitness):
    conflicts = [SimpleNamespace(plan_a=0,plan_b=i+1) for i in range(pairs)]
    return CachedEvaluation(fitness,dict(Nc=points),conflicts)


@pytest.mark.parametrize('parent,trial,accepted',[
    ((3,3,1),(2,9,100),True),
    ((2,9,1),(2,8,100),True),
    ((2,8,100),(2,8,99),True),
    ((2,8,100),(3,3,1),False),
    ((2,8,100),(2,9,1),False),
    ((2,8,100),(2,8,101),False),
    ((2,8,100),(2,8,100),False),
    ((2,8,100),(0,0,float('inf')),False),
])
def test_lexicographic_rules(parent,trial,accepted):
    assert global_conflict_non_deterioration_accept(evaluation(*parent),evaluation(*trial))==accepted


def test_rollback_restores_parent_position_context_and_evaluation():
    parent,context = np.array([1.,2.]),np.array([0,2])
    cached = evaluation(1,1,10)
    position,retained_context,retained_eval,accepted = retain_candidate(parent,context,cached,
        np.array([9.,9.]),np.array([2,1]),evaluation(2,2,1))
    assert not accepted
    np.testing.assert_array_equal(position,parent)
    np.testing.assert_array_equal(retained_context,context)
    assert retained_eval is cached and retained_eval.fitness==10
    position[0]=100
    retained_context[0]=1
    assert parent[0]==1 and context[0]==0


def test_original_fata_golden_path_unchanged():
    result = fata_optimize_paper(lambda x,g:float(np.sum(x*x)),[-1]*3,[1]*3,3,
        population=6,max_iter=4,seed=7,n_jobs=1)
    np.testing.assert_array_equal(result.best_position,[-.10984738823470686]*3)
    np.testing.assert_array_equal(result.convergence,
        [.1966080605919251,.03619934610595924,.03619934610595924,.03619934610595924])


def test_rejected_contexts_do_not_teach_and_parents_are_not_redecoded(monkeypatch):
    from src import global_safeguard as module
    from src.config import load_config
    cfg = load_config()
    cfg['fata'].update(NP=4,Ngen_max_stage2=3)
    cfg['optimization']['n_jobs']=1
    layout = SimpleNamespace(dim=2,lower=np.zeros(2),upper=np.ones(2),actor_genes=slice(0,1))
    calls,learning = [],[]
    def evaluate(position,generation,context):
        calls.append(generation)
        score=float(np.sum(position**2))
        return SimpleNamespace(fitness=score,components=dict(Nc=0,ORISK=score,score=score),conflicts=[],plans=[])
    objective = SimpleNamespace(evaluation=evaluate,reference=None)
    original_update = module.update_probability_matrix
    def update(probability,contexts,rate):
        learning.append(contexts.copy())
        return original_update(probability,contexts,rate)
    monkeypatch.setattr(module,'update_probability_matrix',update)
    monkeypatch.setattr(module,'paper_fitness',lambda comp,*args:comp['score'])
    monkeypatch.setattr(module,'stage2_flight_strategies',lambda *args:{})
    monkeypatch.setattr(module,'retain_candidate',lambda p,c,e,*args:(p.copy(),c.copy(),e,False))
    remaining = [SimpleNamespace(plan_a=1,plan_b=2,cell=(0,0,0),idx_a=0,idx_b=0)]
    result,rows,masks,_ = module.fata_optimize_paper_with_safeguard(objective,layout,cfg,7,remaining)
    assert len(learning)==1  # Only initialization; all subsequent trials rejected.
    assert masks[0].all() and not masks[1:].any()
    assert rows[1]['adm_learning_samples']==rows[2]['adm_learning_samples']==0
    assert len(calls)==4*3+1  # Trials plus final output, no parent decode calls.
    np.testing.assert_array_equal(result.probability_history[1],result.probability_history[-1])
