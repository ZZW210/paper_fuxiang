from __future__ import annotations

from dataclasses import astuple
import inspect

import numpy as np
import pandas as pd
import pytest

from src import fata, optimization_model, adm_matching as adm
from src.config import load_config, apply_quick_overrides
from src.conflict_detection import Conflict, detect_conflicts, IncrementalConflictEvaluator
from src.conflict_network import select_paper_key_flights
from src.flight_plan import FlightPlan, compute_eta_times
from src.grid import AirspaceGrid
from src.paper_optimization import (PaperReference, paper_atd_bounds, paper_objective_components,
    paper_fitness, conflict_weight_delta, build_stage1_decision_layout, build_stage2_decision_layout,
    decode_stage1_solution, decode_stage2_solution, InfeasiblePaperRoute, decode_primary_strategy,
    flight_strategies, flight_strategy_requirements, CandidatePlanView, PaperContributionCache,
    PaperPopulationObjective, RerouteLRU)
from src.paper_scheduler import optimize_paper_schedule


@pytest.fixture
def scenario():
    cfg = load_config("nonexistent.yaml")
    cfg["optimization"]["n_jobs"] = 1
    cfg["fata"].update(NP=6, Ngen_max_stage1=3, Ngen_max_stage2=3)
    grid = AirspaceGrid(shape=(8, 8, 4), obstacles=np.zeros((8, 8, 4), dtype=bool))
    path = [(i, 3, 1) for i in range(1, 7)]
    plans = [FlightPlan(i, path[0], path[-1], list(path), 1000.0,
                       compute_eta_times(path, 1000, 10, grid.cell_size), [10.0] * 5, 6.0, 50.0) for i in range(2)]
    return cfg, grid, np.ones(grid.shape), plans, detect_conflicts(plans, cfg)


def vector(layout, plans):
    x = (layout.lower + layout.upper) / 2
    by_id = {p.id: p for p in plans}
    if layout.actor_genes is not None:
        x[layout.actor_genes] = 0
    for b in layout.blocks:
        x[b.atd_gene] = by_id[b.flight_id].atd
        x[b.speed_genes] = [by_id[b.flight_id].speed_profile[j] for j in b.segment_indices]
        for genes in b.reroute_genes:
            x[genes] = [4, 5, 2]
    return x


@pytest.mark.parametrize("gene,expected", [(0,0),(.999,0),(1,1),(1.999,1),(2,2),(3,2)])
def test_stage1_has_one_specific_strategy(scenario, gene, expected):
    cfg, grid, risk, plans, conflicts = scenario
    layout = build_stage1_decision_layout(plans, [0], conflicts, cfg, grid)
    x = vector(layout, plans)
    b = layout.blocks[0]
    x[b.strategy_gene], x[b.atd_gene], x[b.speed_genes] = gene, 800, 11.374
    decoded = decode_stage1_solution(x, plans, layout, cfg, grid, risk)
    assert decode_primary_strategy(gene) == expected
    assert flight_strategies(decoded[0]) == (expected,)
    assert decoded[0].atd == (800 if expected == 0 else 1000)
    assert all(v == (11.374 if expected == 1 else 10) for v in decoded[0].speed_profile)
    assert decoded[0].rerouted == (expected == 2)
    assert decoded[1] is plans[1]


@pytest.mark.parametrize("atd", [800.123,1200.456])
def test_continuous_atd_and_absolute_delay(scenario, atd):
    cfg, grid, risk, plans, conflicts = scenario
    layout = build_stage1_decision_layout(plans,[0],conflicts,cfg,grid)
    x = vector(layout,plans)
    x[layout.blocks[0].strategy_gene] = 0
    x[layout.blocks[0].atd_gene] = atd
    decoded = decode_stage1_solution(x,plans,layout,cfg,grid,risk)
    c = paper_objective_components(decoded,plans,risk,[],cfg)
    assert c["Tdelay"] == abs(1000-atd)
    assert c["n_delay"] == int(atd>1000)
    assert plans[0].atd == 1000


@pytest.mark.parametrize("etd,expected", [(1,(1,1801)),(1000,(1,2800)),(3500,(1700,3600))])
def test_atd_bounds(scenario, etd, expected):
    cfg,_,_,plans,_ = scenario
    p=plans[0].copy()
    p.etd=etd
    assert paper_atd_bounds(p,cfg)==expected


def test_continuous_segment_speed(scenario):
    cfg,grid,risk,plans,conflicts=scenario
    layout=build_stage1_decision_layout(plans,[0],conflicts,cfg,grid)
    x=vector(layout,plans)
    b=layout.blocks[0]
    x[b.strategy_gene]=1
    x[b.speed_genes.start+2]=11.374
    p=decode_stage1_solution(x,plans,layout,cfg,grid,risk)[0]
    assert p.eta_times[:3]==plans[0].eta_times[:3]
    assert p.eta_times[3]==pytest.approx(p.eta_times[2]+100/11.374)


@pytest.mark.parametrize("actor,changed", [(0,0),(.999,0),(1,1),(2,1)])
def test_actor_is_fata_decision_not_parity(scenario, actor, changed):
    cfg,grid,risk,plans,conflicts=scenario
    layout=build_stage2_decision_layout(plans,conflicts,cfg,grid,plans)
    x=vector(layout,plans)
    x[layout.actor_genes]=actor
    for b in layout.blocks:
        x[b.atd_gene]=800
    final=decode_stage2_solution(x,plans,plans,layout,np.zeros(len(conflicts),int),cfg,grid,risk)
    assert [p.id for p in final if p.changed]==[changed]
    assert final[1-changed] is plans[1-changed]


def test_stage2_can_create_multi_strategy_flight(scenario):
    cfg,grid,risk,plans,_=scenario
    plans=[plans[0].copy(),plans[1].copy(),plans[1].copy()]
    for p,fid in zip(plans,(57,20,74)):
        p.id=fid
    conflicts=[Conflict(57,20,plans[0].path[1],1,1,1010,1010,0,30,"uncertain"),
               Conflict(57,74,plans[0].path[4],4,4,1040,1040,0,30,"uncertain")]
    layout=build_stage2_decision_layout(plans,conflicts,cfg,grid,plans)
    x=vector(layout,plans)
    b=next(b for b in layout.blocks if b.flight_id==57)
    x[b.speed_genes]=11.374
    species=np.array([1,2])
    unchanged=species.copy()
    assert flight_strategy_requirements(layout,species,x[layout.actor_genes])=={57:{1:[0],2:[1]}}
    final=decode_stage2_solution(x,plans,plans,layout,species,cfg,grid,risk)
    assert flight_strategies(final[0])==(1,2)
    assert final[0].rerouted and (4,5,2) in final[0].path
    assert all(v==11.374 for v in final[0].speed_profile)
    assert np.array_equal(species,unchanged)
    assert final[1] is plans[1] and final[2] is plans[2]


def test_stage2_inherits_stage1_speed_and_reroutes(scenario):
    cfg,grid,risk,plans,conflicts=scenario
    l1=build_stage1_decision_layout(plans,[0],conflicts,cfg,grid)
    x=vector(l1,plans)
    x[l1.blocks[0].strategy_gene]=1
    x[l1.blocks[0].speed_genes]=11.374
    current=decode_stage1_solution(x,plans,l1,cfg,grid,risk)
    remaining=detect_conflicts(current,cfg)
    l2=build_stage2_decision_layout(current,remaining,cfg,grid,plans)
    final=decode_stage2_solution(vector(l2,current),current,plans,l2,np.full(len(remaining),2),cfg,grid,risk)
    assert flight_strategies(final[0])==(1,2)
    assert all(v==11.374 for v in final[0].speed_profile)
    assert final[0].rerouted and final[0].atd==1000
    assert final[1] is current[1]
    assert plans[0].speed_profile==[10]*5


def test_stage2_keeps_stage1_atd_when_speed_selected(scenario):
    cfg,grid,risk,plans,_=scenario
    current=[p.copy() for p in plans]
    current[0].atd=990
    current[0].paper_strategies=(0,)
    current[0].eta_times=compute_eta_times(current[0].path,990,10,grid.cell_size)
    remaining=detect_conflicts(current,cfg)
    layout=build_stage2_decision_layout(current,remaining,cfg,grid,plans)
    x=vector(layout,current)
    for b in layout.blocks:
        x[b.speed_genes]=11.374
    final=decode_stage2_solution(x,current,plans,layout,np.ones(len(remaining),int),cfg,grid,risk)
    assert final[0].atd==990 and final[0].delay==-10
    assert flight_strategies(final[0])==(0,1)


def test_local_rerouting_has_no_second_switch_and_is_feasible(scenario):
    cfg,grid,risk,plans,conflicts=scenario
    layout=build_stage2_decision_layout(plans,conflicts,cfg,grid,plans)
    x=vector(layout,plans)
    final=decode_stage2_solution(x,plans,plans,layout,np.full(len(conflicts),2),cfg,grid,risk)
    p=final[0]
    assert (4,5,2) in p.path
    assert p.path[0]==p.start and p.path[-1]==p.goal
    assert len(p.speed_profile)==len(p.path)-1 and all(grid.is_free(c) for c in p.path)
    assert all(max(abs(a[j]-b[j]) for j in range(3))<=1 and a!=b for a,b in zip(p.path,p.path[1:]))
    assert not hasattr(layout.blocks[0],"reroute_enable_genes")
    grid.obstacles[4,5,2]=True
    with pytest.raises(InfeasiblePaperRoute):
        decode_stage2_solution(x,plans,plans,layout,np.full(len(conflicts),2),cfg,grid,risk)


def test_variable_slots_are_deduplicated_and_local(scenario):
    cfg,_,_,plans,_=scenario
    grid=AirspaceGrid(shape=(60,60,4),obstacles=np.zeros((60,60,4),bool))
    plans=[p.copy() for p in plans]
    for p in plans:
        p.path=[(i,30,1) for i in range(26,33)]
        p.start,p.goal=p.path[0],p.path[-1]
        p.speed_profile=[10]*6
        p.eta_times=compute_eta_times(p.path,1000,10,grid.cell_size)
    conflicts=detect_conflicts(plans,cfg)
    layout=build_stage2_decision_layout(plans,conflicts,cfg,grid,plans)
    assert len(layout.blocks)==2
    for b in layout.blocks:
        assert len(b.segment_indices)==len(set(b.segment_indices))
        assert len(b.reroute_genes)==1
        genes=b.reroute_genes[0]
        assert layout.lower[genes].tolist()==[21,25,0]
        assert layout.upper[genes].tolist()==[37,35,2]


def canonical(conflicts):
    return sorted(astuple(c) for c in conflicts)


@pytest.mark.parametrize("uncertain",[True,False])
def test_incremental_matches_full_for_fifty_candidates(scenario,uncertain):
    cfg,grid,_,plans,_=scenario
    base=[]
    for i in range(12):
        p=plans[0].copy()
        p.id=i
        p.atd=1000+19*i
        p.eta_times=compute_eta_times(p.path,p.atd,10,grid.cell_size)
        base.append(p)
    evaluator=IncrementalConflictEvaluator(base,{0,1,2},cfg,uncertain)
    rng=np.random.default_rng(2025)
    for n in range(50):
        view=CandidatePlanView(tuple(base))
        for fid in (0,1,2):
            p=base[fid].copy()
            if n%5==0 and fid==0:
                p.path=p.path[:2]+[(3,4,1),(4,4,1)]+p.path[4:]
            elif n%7==0 and fid==0:
                p.path=p.path[:3]+p.path[1:3]+p.path[3:]
            p.atd=float(rng.uniform(800,1400))
            p.speed_profile=rng.uniform(5,20,len(p.path)-1).tolist()
            p.eta_times=compute_eta_times(p.path,p.atd,p.speed_profile,grid.cell_size)
            view.overrides[fid]=p
        full=detect_conflicts(view,cfg,uncertain)
        incremental=evaluator.evaluate(view)
        assert canonical(full)==canonical(incremental)
        assert [astuple(c) for c in full]==[astuple(c) for c in incremental]


def test_cached_objective_matches_full_fifty_candidates(scenario):
    cfg,grid,risk,plans,_=scenario
    cache=PaperContributionCache(plans,plans,risk,cfg)
    ref=PaperReference.from_initial(plans,risk,[])
    rng=np.random.default_rng(5)
    for _ in range(50):
        p=plans[0].copy()
        p.atd=float(rng.uniform(1,2800))
        p.speed_profile=rng.uniform(5,20,5).tolist()
        p.eta_times=compute_eta_times(p.path,p.atd,p.speed_profile,grid.cell_size)
        view=CandidatePlanView(plans,{0:p})
        conflicts=detect_conflicts(view,cfg)
        full=paper_objective_components(view,plans,risk,conflicts,cfg)
        cached=cache.evaluate(view,conflicts)
        assert cached==full
        assert abs(paper_fitness(cached,ref,cfg,3,3)-paper_fitness(full,ref,cfg,3,3))<1e-10


def test_reroute_lru_persists_and_tracks_environment(scenario):
    cfg,grid,risk,_,_=scenario
    cache=RerouteLRU(maxsize=2)
    cache.bind(grid,risk,cfg)
    first=cache.route((1,3,1),(4,5,2),(6,3,1),grid,risk,cfg)
    assert first==cache.route((1,3,1),(4,5,2),(6,3,1),grid,risk,cfg)
    assert cache.profile.values["astar_cache_hits"]==1
    assert cache.profile.values["astar_calls"]==2
    cache.route((1,3,1),(4,4,2),(6,3,1),grid,risk,cfg)
    cache.route((1,3,1),(4,6,2),(6,3,1),grid,risk,cfg)
    assert len(cache.entries)==2
    grid.obstacles[0,0,0]=True
    cache.bind(grid,risk,cfg)
    cache.route((1,3,1),(4,6,2),(6,3,1),grid,risk,cfg)
    assert cache.profile.values["astar_cache_misses"]==4


@pytest.mark.parametrize("mode",["raw_equation","initial_reference_experimental"])
def test_performance_switches_leave_evaluation_equal(scenario,mode):
    cfg,grid,risk,plans,conflicts=scenario
    cfg["optimization"]["paper_objective_scale_mode"]=mode
    layout=build_stage2_decision_layout(plans,conflicts,cfg,grid,plans)
    ref=PaperReference.from_initial(plans,risk,conflicts)
    optimized=PaperPopulationObjective(plans,plans,layout,cfg,grid,risk,ref,3,2)
    import copy
    other=copy.deepcopy(cfg)
    other["paper_performance"]={"incremental_conflicts":False,"objective_cache":False,"reroute_cache":False}
    full=PaperPopulationObjective(plans,plans,layout,other,grid,risk,ref,3,2)
    for species in (np.zeros(len(conflicts),int),np.ones(len(conflicts),int),np.full(len(conflicts),2)):
        a,b=optimized.evaluation(vector(layout,plans),3,species),full.evaluation(vector(layout,plans),3,species)
        assert a.fitness==b.fitness and a.components==b.components
        assert canonical(a.conflicts)==canonical(b.conflicts)
        assert [p.path for p in a.plans]==[p.path for p in b.plans]


def test_adm_original_sampling_and_probability_stability(scenario,monkeypatch):
    cfg,grid,risk,plans,conflicts=scenario
    cfg["fata"]["NP"]=4
    layout=build_stage2_decision_layout(plans,conflicts,cfg,grid,plans)
    sampled=np.stack([np.arange(len(conflicts))%3,np.full(len(conflicts),2),np.full(len(conflicts),1),np.zeros(len(conflicts),int)])
    before=sampled.copy()
    monkeypatch.setattr(adm,"sample_strategy_species",lambda *args: sampled.copy())
    def fake_fata(objective,lower,upper,dim,**kwargs):
        contexts=kwargs["generation_context"](1,4,np.random.default_rng(1))
        positions=np.stack([vector(layout,plans)]*4)
        kwargs["on_generation_evaluated"](1,positions,np.array([1,3,2,4]),contexts)
        return fata.PaperFATAResult(positions[0],1,[1],[],[],contexts[0].copy(),{},4,1)
    monkeypatch.setattr(fata,"fata_optimize_paper",fake_fata)
    result=adm.adm_fata_optimize(plans,plans,conflicts,layout,cfg,grid,risk,PaperReference.from_initial(plans,risk,conflicts))
    assert np.array_equal(result.best_strategies,sampled[0]) and np.array_equal(sampled,before)
    assert np.array_equal(result.final_probability,adm.update_probability_matrix(np.full((len(conflicts),3),1/3),sampled[[0]],.5))
    p=np.full((1,3),1/3)
    for _ in range(250):
        p=adm.update_probability_matrix(p,np.array([[2]]))
    assert np.all(p>0) and np.allclose(p.sum(axis=1),1) and p[0,0]>=.999e-12


def paper_sphere(x,g):
    return float(np.sum(x*x)*conflict_weight_delta(g,6))


@pytest.mark.parametrize("dim",[3,40,180])
def test_vectorized_update_preserves_seed_bitwise(dim):
    a=fata.fata_optimize_paper(paper_sphere,-2,2,dim,population=8,max_iter=6,n_jobs=1,seed=9,vectorized_update=False)
    b=fata.fata_optimize_paper(paper_sphere,-2,2,dim,population=8,max_iter=6,n_jobs=1,seed=9,vectorized_update=True)
    assert np.array_equal(a.best_position,b.best_position) and a.convergence==b.convergence


def test_parallel_fitness_reproducible():
    a=fata.fata_optimize_paper(paper_sphere,-2,2,3,population=8,max_iter=6,n_jobs=1,seed=9)
    b=fata.fata_optimize_paper(paper_sphere,-2,2,3,population=8,max_iter=6,n_jobs=8,seed=9)
    assert np.array_equal(a.best_position,b.best_position) and a.convergence==b.convergence


@pytest.mark.parametrize("mode",["raw_equation","initial_reference_experimental"])
def test_real_adm_parallel_reproducible(scenario,mode):
    cfg,grid,risk,plans,conflicts=scenario
    cfg["optimization"]["paper_objective_scale_mode"]=mode
    layout=build_stage2_decision_layout(plans,conflicts,cfg,grid,plans)
    ref=PaperReference.from_initial(plans,risk,conflicts)
    a=adm.adm_fata_optimize(plans,plans,conflicts,layout,cfg,grid,risk,ref,seed=27)
    cfg["optimization"]["n_jobs"]=8
    b=adm.adm_fata_optimize(plans,plans,conflicts,layout,cfg,grid,risk,ref,seed=27)
    assert a.convergence==b.convergence and np.array_equal(a.best_decision_vector,b.best_decision_vector)
    assert np.array_equal(a.best_strategies,b.best_strategies) and np.array_equal(a.final_probability,b.final_probability)


@pytest.mark.parametrize("mode",["raw_equation","initial_reference_experimental"])
def test_real_adm_all_performance_switches_equivalent(scenario,mode):
    import copy
    cfg,grid,risk,plans,conflicts=scenario
    cfg["optimization"]["paper_objective_scale_mode"]=mode
    cfg["paper_performance"]={"fata_vectorized_update":True}
    other=copy.deepcopy(cfg)
    other["paper_performance"]={"incremental_conflicts":False,"objective_cache":False,"reroute_cache":False,"fata_vectorized_update":False}
    layout=build_stage2_decision_layout(plans,conflicts,cfg,grid,plans)
    ref=PaperReference.from_initial(plans,risk,conflicts)
    a=adm.adm_fata_optimize(plans,plans,conflicts,layout,cfg,grid,risk,ref,seed=2025)
    b=adm.adm_fata_optimize(plans,plans,conflicts,layout,other,grid,risk,ref,seed=2025)
    assert a.convergence==b.convergence and np.array_equal(a.best_decision_vector,b.best_decision_vector)
    assert np.array_equal(a.best_strategies,b.best_strategies) and np.array_equal(a.final_probability,b.final_probability)


def test_eq49_to51_and_no_hard_caps(scenario):
    cfg,_,_,_,_=scenario
    ref=PaperReference(3600,100,12,6)
    c=dict(Tdelay=200,Tair=120,ORISK=15,Nc=3,n_delay=2,n_battery=1)
    raw=.2*(.25*200+.25*120+.5*15)+.8*.9*3*120+1200
    assert paper_fitness(c,ref,cfg,200,200)==pytest.approx(raw)
    cfg["optimization"].update(conflict_hard_penalty=1e20,delay_count_cap=0,max_changed_flight_ratio=0)
    assert paper_fitness(c,ref,cfg,200,200)==pytest.approx(raw)
    cfg["optimization"]["paper_objective_scale_mode"]="initial_reference_experimental"
    norm=.2*(.25*200/3600+.25*1.2+.5*15/12)+.8*.9*.5*1.2+1200
    assert paper_fitness(c,ref,cfg,200,200)==pytest.approx(norm)
    assert conflict_weight_delta(0,200)==1 and conflict_weight_delta(200,200)==.9


def test_ci_good_points_no_extra_search():
    metrics=pd.DataFrame(dict(flight_id=range(100),collective_influence=range(100),weighted_collective_influence=range(100,0,-1)))
    assert select_paper_key_flights(metrics,100)==list(range(99,89,-1))
    seen=[]
    fata.fata_optimize_paper(lambda x,g: seen.append(x.copy()) or float(np.sum(x*x)),[0,0],[1,1],2,population=4,max_iter=1,n_jobs=1)
    assert np.array_equal(np.array(seen),fata.good_point_set(4,2,np.zeros(2),np.ones(2)))
    source=inspect.getsource(fata.fata_optimize_paper)
    assert ".normal(" not in source and "local_count" not in source


def test_strict_never_repairs_and_runs_fata_twice(scenario,monkeypatch):
    cfg,grid,risk,plans,conflicts=scenario
    def forbidden(*args,**kwargs):
        raise AssertionError("Strict mode called engineering repair")
    for name in ("greedy_time_deconfliction","repair_conflicts","try_apply_action_with_rollback","independent_matching_deconfliction"):
        monkeypatch.setattr(optimization_model,name,forbidden,raising=False)
    calls=[]
    actual=fata.fata_optimize_paper
    def tracked(*args,**kwargs):
        calls.append(kwargs)
        if len(calls)==1:
            layout=build_stage1_decision_layout(plans,[0],conflicts,cfg,grid)
            x=vector(layout,plans)
            x[layout.blocks[0].strategy_gene]=0
            score=args[0](x,kwargs["max_iter"])
            return fata.PaperFATAResult(x,score,[score],[],[],None,{},1,1)
        return actual(*args,**kwargs)
    monkeypatch.setattr(fata,"fata_optimize_paper",tracked)
    result=optimize_paper_schedule(plans,conflicts,[0],cfg,grid,risk,progress=False)
    assert len(calls)==2 and calls[1]["objective_with_context"] is not None
    assert result.adm.final_probability.shape==(len(conflicts),3)


def test_zero_conflicts_skip_stage2(scenario):
    cfg,grid,risk,plans,_=scenario
    result=optimize_paper_schedule([plans[0]],[],[0],cfg,grid,risk,progress=False)
    assert result.adm is None and result.final is result.stage1


def test_quick_only_changes_budget():
    cfg=load_config("nonexistent.yaml")
    quick=apply_quick_overrides(cfg)
    assert quick["optimization"]==cfg["optimization"] and quick["adm"]==cfg["adm"]
    assert quick["fata"]["NP"]==20 and quick["fata"]["Ngen_max_stage1"]==quick["fata"]["Ngen_max_stage2"]==50
