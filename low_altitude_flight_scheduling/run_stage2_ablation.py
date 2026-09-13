"""Leave-one-mechanism-out diagnostics with scoped legacy solver hooks."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')

from contextlib import ExitStack
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_stage2_ab import origins, pair, write_json
from scan_traffic_seeds import prepare_environment, validate_baseline
from src import optimization_model as legacy
from src.astar_3d import astar_path
from src.conflict_detection import Conflict, ConflictSegment, detect_conflicts, write_conflicts_csv
from src.flight_plan import shift_plan_time, apply_speed_factor
from src.paper_scheduler import _physically_changed
from src.paper_optimization import (PaperReference, paper_objective_components, paper_fitness,
    build_stage1_decision_layout, PaperPopulationObjective)
from src.fata import fata_optimize_paper
from src.run_archive import git_metadata

VARIANTS = ('B0_full', 'B1_no_segment_grouping',
            'B2_no_conflict_informed_candidates', 'B3_no_global_rollback')
SOURCE = '20260913_113232_490610_stage2_ab'


def raw_units(conflicts):
    return [ConflictSegment(c.plan_a, c.plan_b, [c]) for c in conflicts]


def fixed_delay(plans, segment, cfg, limit):
    by_id = {p.id:p for p in plans}
    opt = cfg['optimization']
    threshold = float(opt['delay_count_threshold'])
    delayed = legacy.delayed_count(plans, threshold)
    values = sorted(legacy._delay_candidates(cfg), key=lambda x:(abs(x), x))
    actions = []
    for delta in values:
        for fid in (segment.plan_a, segment.plan_b):
            p = by_id[fid]
            if abs(delta)<1 or (delta>0 and p.delay<threshold and delayed>=opt['delay_count_cap']):
                continue
            if p.etd+delta<=0 or p.delay+delta>opt['t_delay_max']:
                continue
            actions.append(dict(type='delay', plan_id=fid, delta=float(delta)))
    return actions[:limit]


def fixed_speed(plans, segment, cfg, limit):
    by_id = {p.id:p for p in plans}
    lo, hi = cfg['optimization']['speed_range']
    actions = []
    for factor in cfg['optimization']['speed_factors']:
        for fid in (segment.plan_a, segment.plan_b):
            p = by_id[fid]
            speed = float(np.mean(p.speed_profile or [cfg['flight']['default_speed']]))
            if abs(np.clip(speed*factor,lo,hi)-speed)>1e-6:
                actions.append(dict(type='speed', plan_id=fid, factor=float(factor)))
    return actions[:limit]


def local_improved(old, new, target):
    return sum(pair(c)==target for c in new)<sum(pair(c)==target for c in old)


def run_variant(base, remaining, cfg, grid, risk, seed, variant):
    target = [None]
    counts = dict(number_of_conflict_detections=0, number_of_astar_calls=0)
    original_apply = legacy.try_apply_action_with_rollback
    original_classify = legacy._classify_adm_segment
    trace = []
    last = {}

    def classify(plans, conflicts, segment, *args):
        target[0] = tuple(sorted((segment.plan_a,segment.plan_b)))
        return original_classify(plans,conflicts,segment,*args)

    def detector(plans, *args, **kwargs):
        counts['number_of_conflict_detections'] += 1
        result = detect_conflicts(plans,*args,**kwargs)
        last.update(plans=plans, conflicts=result)
        return result

    def astar(*args, **kwargs):
        counts['number_of_astar_calls'] += 1
        kwargs.update(alpha_r=.8, alpha_l=.2, risk_weight=.8, distance_weight=.2,
                      distance_unit_m=1., vertical_move_penalty=1.)
        return astar_path(*args,**kwargs)

    def apply(plans, action, old, cfg, grid, risk):
        last.clear()
        if variant!='B3_no_global_rollback':
            accepted, trial, new, log = original_apply(plans,action,old,cfg,grid,risk)
            evaluated = last.get('conflicts', old)
        else:
            started = time.perf_counter()
            trial = list(plans)
            idx = next(i for i,p in enumerate(trial) if p.id==action['plan_id'])
            kind = action['type']
            if kind=='delay':
                replacement = shift_plan_time(trial[idx],action['delta'])
            elif kind=='speed':
                replacement = apply_speed_factor(trial[idx],action['factor'],
                    *cfg['optimization']['speed_range'],grid.cell_size)
            else:
                replacement = action['plan'].copy()
            trial[idx] = legacy.recompute_plan(replacement,grid,risk)
            evaluated = detector(trial,cfg,uncertain=True)
            accepted = local_improved(old,evaluated,target[0])
            log = legacy._action_log(kind,action['plan_id'],legacy.count_conflict_pairs(old),
                legacy.count_conflict_pairs(evaluated),len(old),len(evaluated),accepted,started,
                'local_pair_points_decreased' if accepted else 'local_pair_not_improved')
            new = evaluated if accepted else old
            if not accepted:
                trial = plans
        log.update(old_target_points=sum(pair(c)==target[0] for c in old),
            new_target_points=sum(pair(c)==target[0] for c in evaluated),
            new_pairs_created_by_action=len(set(map(pair,evaluated))-set(map(pair,old))))
        trace.append(log)
        return accepted,trial,new,log

    start = time.perf_counter()
    with ExitStack() as stack:
        for name, replacement in dict(astar_path=astar, detect_conflicts=detector,
            _classify_adm_segment=classify, try_apply_action_with_rollback=apply).items():
            stack.enter_context(patch.object(legacy,name,replacement))
        if variant=='B1_no_segment_grouping':
            stack.enter_context(patch.object(legacy,'group_continuous_conflicts',raw_units))
        if variant=='B2_no_conflict_informed_candidates':
            stack.enter_context(patch.object(legacy,'_adm_delay_actions',fixed_delay))
            stack.enter_context(patch.object(legacy,'_adm_speed_actions',fixed_speed))
        final, _, logs = legacy.independent_matching_deconfliction(base,remaining,cfg,grid,risk,
                                                                  max_rounds=60,seed=seed)
    runtime = time.perf_counter()-start
    conflicts = detector(final,cfg,uncertain=True)
    normalized = []
    for log in logs:
        a,b = log.get('pair','-').split('-')
        normalized.append(dict(log, round=log.get('iter'),unit_id=log.get('segment_idx'),
            plan_a=a,plan_b=b,strategy=log.get('matched_strategy'),actor_flight=log.get('plan_id'),
            old_global_points=log.get('old_points'),new_global_points=log.get('new_points'),
            old_global_pairs=log.get('old_pairs'),new_global_pairs=log.get('new_pairs')))
    return final,conflicts,normalized,trace,counts,runtime


def summarize(rows,out):
    frame = pd.DataFrame(rows)
    for name in ('raw_runs.csv','stage2_mechanism_ablation_raw.csv'):
        frame.to_csv(out/name,index=False)
    summary = frame.groupby('variant').agg(n_runs=('final_conflicts','size'),
        final_conflicts_mean=('final_conflicts','mean'),final_conflicts_std=('final_conflicts','std'),
        final_conflicts_median=('final_conflicts','median'),zero_conflict_success_rate=('zero_conflict','mean'),
        new_conflict_pairs_mean=('new_conflict_pairs','mean'),exact_survivors_mean=('exact_survivors','mean'),
        runtime_mean=('runtime_sec','mean'),risk_increase_mean=('risk_increase_percent','mean'),
        changed_flights_mean=('changed_flights','mean'),delayed_flights_mean=('delayed_flights','mean'))
    summary.to_csv(out/'stage2_mechanism_ablation_summary.csv')
    baseline = summary.loc[VARIANTS[0]]
    contributions = []
    for mechanism,variant in zip(('segment_grouping','conflict_informed_candidates','global_rollback'),VARIANTS[1:]):
        row = summary.loc[variant]
        contributions.append(dict(mechanism=mechanism,removed_variant=variant,
            zero_conflict_rate_drop=baseline.zero_conflict_success_rate-row.zero_conflict_success_rate,
            mean_final_conflict_increase=row.final_conflicts_mean-baseline.final_conflicts_mean,
            mean_new_pair_increase=row.new_conflict_pairs_mean-baseline.new_conflict_pairs_mean,
            mean_exact_survivor_increase=row.exact_survivors_mean-baseline.exact_survivors_mean))
    contribution = pd.DataFrame(contributions).sort_values(['zero_conflict_rate_drop',
        'mean_final_conflict_increase','mean_new_pair_increase','mean_exact_survivor_increase'],ascending=False)
    keys = ['zero_conflict_rate_drop','mean_final_conflict_increase','mean_new_pair_increase','mean_exact_survivor_increase']
    ranks = {}; last = None
    for i,(_,r) in enumerate(contribution.iterrows(),1):
        key = tuple(r[k] for k in keys)
        if key!=last:
            rank=i
        ranks[r.mechanism]=rank
        last=key
    contribution['rank']=contribution.mechanism.map(ranks)
    contribution.to_csv(out/'stage2_mechanism_contribution.csv',index=False)
    return summary,contribution


def main():
    root = Path(__file__).resolve().parent
    branch = subprocess.check_output(['git','-c',f'safe.directory={root.parent.as_posix()}',
        'branch','--show-current'],cwd=root.parent,text=True).strip()
    if branch!='scheduling':
        raise RuntimeError(f'Requires scheduling branch, got {branch}')
    source = root/'outputs'/'stage2_ab_test'/SOURCE
    source_manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if source_manifest['status']!='completed':
        raise RuntimeError('Source A/B test not completed')
    cfg = source_manifest['config']
    validate_baseline(cfg)
    grid,risk,hashes = prepare_environment(cfg,2025)
    if hashes!=source_manifest['environment_hashes']:
        raise RuntimeError('Environment hash mismatch')
    initial = pickle.loads((source/'shared'/'initial_plans.pkl').read_bytes())
    initial_conflicts = detect_conflicts(initial,cfg,uncertain=True)
    reference = PaperReference.from_initial(initial,risk,initial_conflicts,cfg['optimization']['t_delay_max'])
    risk0 = paper_objective_components(initial,initial,risk,initial_conflicts,cfg)['ORISK']
    test_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_ablation'
    out = root/'outputs'/'stage2_mechanism_ablation'/test_id
    out.mkdir(parents=True,exist_ok=False)
    commit,dirty = git_metadata(root.parent)
    manifest = dict(status='validating',branch=branch,git_commit=commit,git_dirty=dirty,
        source=str(source),config=cfg,environment_hashes=hashes,stage1_runs=0,
        local_acceptance='B3 strictly reduces total target pair conflict points; global counts never veto',
        expansion_rule='Expand to seeds 3..9 if any two ablations have zero-rate drops equal and final-conflict mean differences <=1, or all zero',
        detection_count_scope='All solver action detections plus final validation; excludes shared input checks',
        source_code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def checkpoint():
        write_json(out/'manifest.json',manifest)
    checkpoint()
    shared = {}
    for seed,expected in enumerate((51,54,52)):
        directory = source/f'seed_{seed}'
        blob = (directory/'shared_stage1_plans.pkl').read_bytes()
        old = detect_conflicts(pickle.loads(blob),cfg,uncertain=True)
        saved = pd.read_csv(directory/'shared_stage1_conflicts.csv')
        digest = hashlib.sha256(blob+pickle.dumps(old)).hexdigest()
        meta = json.loads((directory/'shared_stage1_metrics.json').read_text())
        # Check saved CSV fields against the regenerated exact events, not just count.
        check = out/f'validation_seed_{seed}.csv'
        write_conflicts_csv(old,check)
        try:
            pd.testing.assert_frame_equal(saved,pd.read_csv(check))
        except AssertionError:
            manifest['status']='stopped_shared_conflict_csv_mismatch'; checkpoint(); raise
        if len(old)!=expected or digest!=meta['input_hash']:
            manifest['status']='stopped_shared_input_mismatch'; checkpoint()
            raise RuntimeError(f'Seed {seed}: count/hash mismatch')
        shared[seed]=(blob,pickle.dumps(old),digest)
    rows = []
    def execute(seed,variant):
        blob,cblob,digest = shared[seed]
        base,remaining = pickle.loads(blob),pickle.loads(cblob)
        final,conflicts,logs,actions,counts,runtime = run_variant(base,remaining,cfg,grid,risk,seed,variant)
        assert hashlib.sha256(pickle.dumps(base)+pickle.dumps(remaining)).hexdigest()==digest
        comp = paper_objective_components(final,initial,risk,conflicts,cfg)
        accepted = [a for a in actions if a['accepted']]
        row = dict(optimizer_seed=seed,variant=variant,stage1_input_hash=digest,
            stage1_conflicts=len(remaining),final_conflicts=len(conflicts),
            final_conflict_pairs=legacy.count_conflict_pairs(conflicts),
            conflict_reduction_ratio=(len(remaining)-len(conflicts))/max(1,len(remaining)),
            zero_conflict=len(conflicts)==0,**origins(remaining,conflicts),
            changed_flights=sum(_physically_changed(a,b) for a,b in zip(initial,final)),
            delayed_flights=comp['n_delay'],risk_increase_percent=100*(comp['ORISK']-risk0)/risk0,
            final_paper_fitness=paper_fitness(comp,reference,cfg,200,200),runtime_sec=runtime,
            accepted_actions=len(accepted),rejected_actions=len(actions)-len(accepted),
            rollback_count=sum(a['rollback'] for a in actions),
            delay_actions_accepted=sum(a['action_type']=='delay' for a in accepted),
            speed_actions_accepted=sum(a['action_type']=='speed' for a in accepted),
            reroute_actions_accepted=sum(a['action_type'] in ('reroute','cruise_level') for a in accepted),**counts)
        directory = out/f'seed_{seed}'/variant
        directory.mkdir(parents=True)
        pd.DataFrame(logs).to_csv(directory/'action_trace.csv',index=False)
        write_json(directory/'metrics.json',row)
        write_conflicts_csv(conflicts,directory/'final_conflicts.csv')
        (directory/'final_plans.pkl').write_bytes(pickle.dumps(final))
        rows.append(row)
        pd.DataFrame(rows).to_csv(out/'raw_runs.csv',index=False)
        print(f'seed={seed} {variant} final={len(conflicts)} accepted={len(accepted)} runtime={runtime:.2f}s',flush=True)
        return row
    manifest['status']='sanity_check'; checkpoint()
    for seed in (0,1,2):
        row = execute(seed,VARIANTS[0])
        if row['final_conflicts']!=0:
            manifest['status']='stopped_B0_reproducibility_failure';checkpoint();return 1
        previous = pickle.loads((source/f'seed_{seed}'/'case_B_legacy_targeted_stage2'/'final_plans.pkl').read_bytes())
        assert all(not _physically_changed(a,b) for a,b in zip(previous,
            pickle.loads((out/f'seed_{seed}'/VARIANTS[0]/'final_plans.pkl').read_bytes())))
    manifest['status']='ablating';checkpoint()
    for seed in (0,1,2):
        for variant in VARIANTS[1:]:
            execute(seed,variant)
    summary,contribution = summarize(rows,out)
    ab = summary.loc[list(VARIANTS[1:])]
    close = any(abs(ab.iloc[i].zero_conflict_success_rate-ab.iloc[j].zero_conflict_success_rate)<1e-9
        and abs(ab.iloc[i].final_conflicts_mean-ab.iloc[j].final_conflicts_mean)<=1 for i in range(3) for j in range(i))
    manifest['expanded']=bool(close)
    if close:
        keys = json.loads((source/'shared'/'key_ids.json').read_text())
        for seed in range(3,10):
            layout = build_stage1_decision_layout(initial,keys,initial_conflicts,cfg,grid)
            objective = PaperPopulationObjective(initial,initial,layout,cfg,grid,risk,reference,200,1)
            result = fata_optimize_paper(objective.fitness,layout.lower,layout.upper,layout.dim,
                population=50,max_iter=200,seed=seed,parf=cfg['fata']['Parf'],n_jobs=cfg['optimization']['n_jobs'],
                vectorized_update=cfg['paper_performance']['fata_vectorized_update'])
            if not np.isfinite(result.best_fitness):
                raise RuntimeError('Extended Stage1 infeasible')
            stage1 = objective.evaluation(result.best_position,200)
            blob,cblob = pickle.dumps(list(stage1.plans)),pickle.dumps(stage1.conflicts)
            digest = hashlib.sha256(blob+cblob).hexdigest()
            shared[seed]=(blob,cblob,digest)
            directory = out/f'seed_{seed}'
            directory.mkdir()
            (directory/'shared_stage1_plans.pkl').write_bytes(blob)
            write_conflicts_csv(stage1.conflicts,directory/'shared_stage1_conflicts.csv')
            write_json(directory/'shared_stage1_metrics.json',dict(input_hash=digest,**stage1.components))
            manifest['stage1_runs']+=1;checkpoint()
            for variant in VARIANTS:
                execute(seed,variant)
        summary,contribution = summarize(rows,out)
    best = contribution.iloc[0]
    tied = contribution[contribution['rank']==1]
    proposals = dict(segment_grouping='preprocessing/grouping with shared regional variables',
        conflict_informed_candidates='FATA warm start or candidate initialization, not replacing FATA',
        global_rollback='soft feasibility safeguard or elite preservation; extra undisclosed constraint, never insert into paper_strict')
    report = ['# Stage2 Mechanism Ablation Report',f'Test: {test_id}; paired seeds: {len(shared)}; extended Stage1 runs: {manifest["stage1_runs"]}.',
        'B0 reproduced the previous 0/0/0 and physically identical final plans before ablations.',
        '## Results','```text\n'+summary.to_string()+'\n```','## Contributions (lexicographic)',
        '```text\n'+contribution.to_string(index=False)+'\n```',
        'Rate drop is a fraction; multiply by 100 for percentage points. No weighted score. Tied lexicographic values share ranks.',
        '## Answers',
        f'Largest zero-rate drop: {contribution.loc[contribution.zero_conflict_rate_drop==contribution.zero_conflict_rate_drop.max(),"mechanism"].tolist()}.',
        f'Largest mean conflict increase: {contribution.loc[contribution.mean_final_conflict_increase==contribution.mean_final_conflict_increase.max(),"mechanism"].tolist()}.',
        f'Largest final new-pair increase (suppression evidence): {contribution.loc[contribution.mean_new_pair_increase==contribution.mean_new_pair_increase.max(),"mechanism"].tolist()}.',
        f'Largest exact survivor increase: {contribution.loc[contribution.mean_exact_survivor_increase==contribution.mean_exact_survivor_increase.max(),"mechanism"].tolist()}.',
        'Leave-one-out effects are conditional on the other two mechanisms. Interactions are possible, but cannot be established or quantified without factorial experiments.',
        f'First A+ candidate: {best.mechanism if len(tied)==1 else "not uniquely distinguishable: "+str(tied.mechanism.tolist())}.',
        proposals[best.mechanism],
        '## Definitions and Controls',
        'B1 uses one raw Conflict per unit in detector order, never calls grouping. Classification stays unchanged. B2 fixed delay order=(absolute value, signed value), alternating a/b per value; speed follows configured factor order alternating a/b. Original eligibility filters and limits stay fixed. Reroute geometry and actor order remain unchanged and do not inspect arrival time.',
        'B3 accepts strictly reduced total target-pair points, including pair disappearance; global detection does not veto. First accepted action behavior stays fixed. Round processing still stops on no accepted improvement, maximum 60.',
        'Action trace new_pairs_created_by_action includes rejected hypothetical candidates. Accepted/rejected counts exclude no_candidate and unresolved notices. Reroute accepted includes altitude shifts. Detection counts include final validation; A* counts actual calls including unsuccessful searches.',
        'Origins use prior A/B definitions: exact pair/cell/aligned indices ignoring arrival times; moved same pair counts nonexact events on original pairs; new pairs counts final distinct pairs absent from shared Stage1. Resolved original includes moved events. Runtime solver only; risk, changes and delay relative to initial scene. All current A* parameters retained through the same adapter as B0.',
        'No A or formal paper_strict code modified. Finite paired sample results are descriptive, not statistical significance or standalone mechanism sufficiency claims.']
    (out/'stage2_mechanism_ablation_report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    manifest['status']='completed';checkpoint()
    print(out,flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
