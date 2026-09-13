from types import SimpleNamespace
import numpy as np
import pytest

from src import conflict_elite as module
from src.conflict_elite import ConflictEliteArchive,Evaluation,key,run_diagnostic_fata
from src.config import load_config
from src.paper_optimization import paper_fitness
from src.fata import fata_optimize_paper


def evaluation(pairs,points,score):
    return Evaluation(score,dict(Nc=points,score=score,n_delay=0,ORISK=score),
        [SimpleNamespace(plan_a=0,plan_b=i+1) for i in range(pairs)],0)


@pytest.fixture
def archive(monkeypatch):
    monkeypatch.setattr(module,'paper_fitness',lambda comp,*args:comp['score'])
    objective=SimpleNamespace(reference=None,cfg={},max_gen=200,
        evaluation=lambda *args:SimpleNamespace(plans=[{'independent':True}]))
    return ConflictEliteArchive(objective)


@pytest.mark.parametrize('parent,trial,better',[
    ((3,3,1),(2,5,10),True),((2,5,1),(2,4,10),True),
    ((2,4,10),(2,4,9),True),((2,4,10),(3,1,1),False),
    ((2,4,10),(2,5,1),False),((2,4,10),(2,4,11),False),
    ((2,4,10),(2,4,10),False)])
def test_archive_updates_by_strict_lexicographic_key(archive,parent,trial,better):
    archive.consider(np.array([0.]),np.array([0]),evaluation(*parent),1)
    assert archive.consider(np.array([1.]),np.array([1]),evaluation(*trial),1)==better
    assert key(archive.evaluation)==key(evaluation(*(trial if better else parent)))


def test_initialization_selects_lex_best_not_lowest_fitness(archive):
    entries=[evaluation(3,3,1),evaluation(2,5,9),evaluation(2,4,10)]
    for i,e in enumerate(entries):archive.consider(np.array([float(i)]),np.array([i]),e,1)
    assert key(archive.evaluation)==(2,4,10)


def test_present_requires_both_position_and_context_and_no_duplicate(archive):
    archive.consider(np.array([1.]),np.array([2]),evaluation(1,1,1),1)
    positions=np.array([[1.],[2.]]);contexts=np.array([[2],[0]])
    assert archive.inject(positions,contexts,[evaluation(1,1,1),evaluation(2,2,2)]) is None
    contexts[0]=1
    assert not archive.present(positions,contexts)


def test_missing_elite_replaces_worst_with_independent_full_state(archive):
    archive.consider(np.array([1.]),np.array([2]),evaluation(1,1,1),1)
    positions=np.array([[3.],[4.],[5.]])
    contexts=np.array([[0],[0],[0]])
    evaluations=[evaluation(2,2,100),evaluation(3,3,1),evaluation(2,5,1)]
    injected=archive.inject(positions,contexts,evaluations)
    assert injected[0]==1
    np.testing.assert_array_equal(positions[1],archive.position)
    np.testing.assert_array_equal(contexts[1],archive.context)
    assert evaluations[1].components==archive.evaluation.components
    assert key(evaluations[1])==key(archive.evaluation)
    positions[1]=999;contexts[1]=0;evaluations[1].components['Nc']=999
    assert archive.position[0]==1 and archive.context[0]==2 and archive.evaluation.components['Nc']==1
    assert not archive.consider(np.array([999.]),np.array([0]),evaluation(99,99,0),2)


def test_historical_elite_reprices_with_real_delta_without_decode():
    cfg=load_config()
    comp=dict(Nc=2,Tdelay=0,Tair=100,ORISK=1,n_delay=0,n_battery=0)
    calls=[]
    objective=SimpleNamespace(reference=None,cfg=cfg,max_gen=200,
        evaluation=lambda *args:(calls.append(1) or SimpleNamespace(plans=[])))
    archive=ConflictEliteArchive(objective)
    e=evaluation(2,2,paper_fitness(comp,None,cfg,1,200));e.components=comp
    archive.consider(np.array([1.]),np.array([0]),e,1)
    score=archive.evaluation.fitness
    archive.reprice(200)
    assert len(calls)==1 and archive.evaluation.fitness<score
    assert archive.evaluation.fitness==paper_fitness(comp,None,cfg,200,200)


def test_diagnostic_no_elite_matches_original_and_adm_has_no_mask():
    cfg=load_config()
    class Objective:
        initial_plans=[];reference=None;max_gen=4
        def __init__(self):self.cfg=cfg
        def evaluation(self,x,g,c):
            comp=dict(Nc=0,Tdelay=0,Tair=float(np.sum(x*x)),ORISK=0,n_delay=0,n_battery=0)
            return SimpleNamespace(fitness=paper_fitness(comp,None,cfg,g,4),components=comp,conflicts=[],plans=[])
        def fitness(self,x,g):return self.evaluation(x,g,None).fitness
        def context_fitness(self,x,g,c):return self.evaluation(x,g,c).fitness
    objective=Objective();learned=[]
    def contexts(g,n,rng):return rng.integers(0,3,(n,1))
    def update(g,positions,fitness,contexts):learned.append((g,len(fitness),len(contexts)))
    args=dict(population=6,max_iter=4,seed=7,n_jobs=1,
        generation_context=contexts,objective_with_context=objective.context_fitness)
    original=fata_optimize_paper(objective.fitness,[-1]*3,[1]*3,3,**args)
    records={}
    result=run_diagnostic_fata(objective.fitness,[-1]*3,[1]*3,3,
        **args,on_generation_evaluated=update,records=records,enable_elite=False)
    np.testing.assert_array_equal(result.best_position,original.best_position)
    np.testing.assert_array_equal(result.best_context,original.best_context)
    np.testing.assert_array_equal(result.convergence,original.convergence)
    assert learned==[(g,6,6) for g in range(1,5)]
    assert records['reinjections']==0 and records['ever_zero']


def test_generation_has_no_parent_trial_rollback():
    import inspect
    source=inspect.getsource(run_diagnostic_fata)
    assert 'retain_candidate' not in source and 'accepted_mask' not in source
    assert 'global_conflict_non_deterioration_accept' not in source
