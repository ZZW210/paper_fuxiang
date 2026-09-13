"""Paired diagnostic experiment; does not modify either formal solver."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')

from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd

from scan_traffic_seeds import prepare_environment, validate_baseline
from src.config import load_config, resolve_scene_seeds
from src.conflict_detection import detect_conflicts, write_conflicts_csv, count_conflict_pairs
from src.conflict_network import build_conflict_network, network_metrics, select_paper_key_flights
from src.flight_plan import sample_paper_random_traffic, plan_paper_random_traffic
from src.fata import fata_optimize_paper
from src.adm_matching import adm_fata_optimize
from src.optimization_model import independent_matching_deconfliction
from src import optimization_model as legacy
from src.astar_3d import astar_path
from src.paper_optimization import (PaperReference, PaperPopulationObjective,
    build_stage1_decision_layout, build_stage2_decision_layout,
    paper_objective_components, paper_fitness)
from src.paper_scheduler import _physically_changed
from src.run_archive import git_metadata


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def pair(c):
    return tuple(sorted((c.plan_a, c.plan_b)))


def event(c):
    indices = (c.idx_a, c.idx_b) if c.plan_a <= c.plan_b else (c.idx_b, c.idx_a)
    return (*pair(c), tuple(c.cell), *indices)


def origins(original, final):
    from collections import Counter
    old = Counter(map(event, original))
    new = Counter(map(event, final))
    old_pairs = set(map(pair, original))
    exact = sum((old & new).values())
    new_points = sum(c not in old_pairs for c in map(pair, final))
    return dict(exact_survivors=exact,
                moved_same_pair=len(final)-exact-new_points,
                new_conflict_pairs=len(set(map(pair, final))-old_pairs),
                new_conflict_points=new_points,
                resolved_original_conflicts=len(original)-exact)


def main():
    root = Path(__file__).resolve().parent
    cfg = load_config(root / 'config.yaml')
    resolve_scene_seeds(cfg, environment_seed=2025, traffic_seed=316)
    validate_baseline(cfg)
    assert (cfg['fata']['NP'], cfg['fata']['Ngen_max_stage1'], cfg['fata']['Ngen_max_stage2']) == (50, 200, 200)
    expected_legacy = dict(independent_matching_rounds=60, independent_matching_segment_limit=24,
        independent_matching_candidates_per_strategy=20, independent_matching_lrate=.5,
        local_reroute_windows=[8,12], local_reroute_radii=[1,2], safety_margin_seconds=10,
        independent_matching_strategy_priors=dict(head_to_head=[.15,.15,.70],
            cluster=[.50,.30,.20], crossing=[.25,.60,.15]),
        delay_candidates=[-600,-420,-300,-240,-180,-120,-90,-60,-30,0,30,60,90,120,180,240,300,420,600],
        speed_factors=[.90,.95,1.,1.05,1.10,1.15])
    if any(cfg['optimization'].get(k) != v for k,v in expected_legacy.items()):
        raise ValueError('Legacy diagnostic budget/candidates differ from the fixed 106 configuration')
    if (cfg['risk']['alpha_r'], cfg['risk']['alpha_L'], cfg['adm']['learning_rate']) != (.8,.2,.5):
        raise ValueError('Risk weights or ADM learning rate differ from the fixed experiment')
    commit, dirty = git_metadata(root.parent)
    import subprocess
    branch = subprocess.check_output(['git', '-c', f'safe.directory={root.parent.as_posix()}',
                                     'branch', '--show-current'], cwd=root.parent, text=True).strip()
    if branch != 'conflict-network':
        raise RuntimeError(f'Unexpected branch: {branch}')
    test_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_stage2_ab'
    out = root / 'outputs' / 'stage2_ab_test' / test_id
    shared = out / 'shared'
    shared.mkdir(parents=True, exist_ok=False)
    manifest = dict(test_id=test_id, git_commit=commit, git_dirty=dirty,
                    branch=branch, status='validating', config=cfg,
                    legacy_provenance='106-result commit 11e1a04; 6b39685 only added paper API imports; solver logic unchanged',
                    seed_policy='Stage1 and both Stage2 solvers receive the paired optimizer seed directly')
    write_json(out / 'manifest.json', manifest)
    grid, risk, hashes = prepare_environment(cfg, 2025)
    initial = plan_paper_random_traffic(grid, risk, cfg,
                 sample_paper_random_traffic(grid, cfg, n_flights=100, seed=316))
    conflicts = detect_conflicts(initial, cfg, uncertain=True)
    det = detect_conflicts(initial, cfg, uncertain=False)
    manifest.update(environment_hashes=hashes, initial_deterministic=len(det), initial_uncertain=len(conflicts))
    if (len(det), len(conflicts)) != (53, 107):
        manifest['status'] = 'stopped_scene_mismatch'
        write_json(out / 'manifest.json', manifest)
        print(manifest, flush=True)
        return 1
    write_conflicts_csv(det, shared / 'initial_deterministic.csv')
    write_conflicts_csv(conflicts, shared / 'initial_uncertain.csv')
    (shared / 'initial_plans.pkl').write_bytes(pickle.dumps(initial))
    keys = select_paper_key_flights(network_metrics(build_conflict_network(initial, conflicts)), 100, .1)
    write_json(shared / 'key_ids.json', keys)
    reference = PaperReference.from_initial(initial, risk, conflicts, cfg['optimization']['t_delay_max'])
    manifest['status'] = 'running'
    write_json(out / 'manifest.json', manifest)
    rows = []
    for seed in (0, 1, 2):
        seed_dir = out / f'seed_{seed}'
        seed_dir.mkdir()
        layout = build_stage1_decision_layout(initial, keys, conflicts, cfg, grid)
        objective = PaperPopulationObjective(initial, initial, layout, cfg, grid, risk, reference, 200, 1)
        start = time.perf_counter()
        def progress(position, score, generation, context):
            if generation % 20 == 0:
                print(f'seed={seed} Stage1 generation={generation} fitness={score}', flush=True)
            return {}
        result = fata_optimize_paper(objective.fitness, layout.lower, layout.upper, layout.dim,
                    population=50, max_iter=200, seed=seed, parf=cfg['fata']['Parf'],
                    n_jobs=cfg['optimization']['n_jobs'], callback=progress,
                    vectorized_update=cfg['paper_performance']['fata_vectorized_update'])
        if not np.isfinite(result.best_fitness):
            raise RuntimeError('Stage1 has no feasible candidate')
        stage1 = objective.evaluation(result.best_position, 200)
        elapsed = time.perf_counter()-start
        plans_blob = pickle.dumps(list(stage1.plans))
        conflict_blob = pickle.dumps(stage1.conflicts)
        digest = hashlib.sha256(plans_blob+conflict_blob).hexdigest()
        (seed_dir / 'shared_stage1_plans.pkl').write_bytes(plans_blob)
        write_conflicts_csv(stage1.conflicts, seed_dir / 'shared_stage1_conflicts.csv')
        write_json(seed_dir / 'shared_stage1_metrics.json', dict(optimizer_seed=seed,
                   input_hash=digest, runtime_sec=elapsed, fitness=stage1.fitness, **stage1.components))
        print(f'seed={seed} Stage1 Nc={len(stage1.conflicts)}; shared hash={digest}', flush=True)
        for method in ('A_current_stage2', 'B_legacy_targeted_stage2'):
            directory = seed_dir / f'case_{method}'
            directory.mkdir()
            base, remaining = pickle.loads(plans_blob), pickle.loads(conflict_blob)
            assert hashlib.sha256(pickle.dumps(base)+pickle.dumps(remaining)).hexdigest() == digest
            start = time.perf_counter()
            trace = []
            if method.startswith('A') and remaining:
                layout2 = build_stage2_decision_layout(base, remaining, cfg, grid, initial)
                adm = adm_fata_optimize(base, initial, remaining, layout2, cfg, grid, risk,
                                       reference, seed=seed)
                final = list(adm.best_plans)
                trace = [dict(generation=i, fitness=f) for i, f in enumerate(adm.convergence, 1)]
                assignments = [dict(conflict_id=i, plan_a=c.plan_a, plan_b=c.plan_b,
                    strategy=int(adm.best_strategies[i]), actor_gene=float(adm.best_decision_vector[i]),
                    actor_flight=c.plan_a if adm.best_decision_vector[i]<1 else c.plan_b)
                    for i, c in enumerate(remaining)]
                pd.DataFrame(assignments).to_csv(directory / 'stage2_strategy_assignment.csv', index=False)
            elif method.startswith('B') and remaining:
                def controlled_astar(*args, **kwargs):
                    kwargs.update(alpha_r=.8, alpha_l=.2, risk_weight=.8, distance_weight=.2,
                                  distance_unit_m=1.0, vertical_move_penalty=1.0)
                    return astar_path(*args, **kwargs)
                previous_astar = legacy.astar_path
                legacy.astar_path = controlled_astar
                try:
                    final, _, actions = independent_matching_deconfliction(base, remaining, cfg, grid, risk,
                                                                            max_rounds=60, seed=seed)
                finally:
                    legacy.astar_path = previous_astar
                normalized = []
                for action in actions:
                    a, b = action.get('pair', '-').split('-')
                    normalized.append(dict(action, round=action.get('iter'), segment_id=action.get('segment_idx'),
                        plan_a=a, plan_b=b, strategy=action.get('matched_strategy'),
                        actor_flight=action.get('plan_id'), old_conflict_points=action.get('old_points'),
                        new_conflict_points=action.get('new_points'), old_conflict_pairs=action.get('old_pairs'),
                        new_conflict_pairs=action.get('new_pairs')))
                pd.DataFrame(normalized).to_csv(directory / 'stage2_action_trace.csv', index=False)
                trace = [a for a in normalized if a.get('accepted')]
            else:
                final = base
            runtime = time.perf_counter()-start
            final_conflicts = detect_conflicts(final, cfg, uncertain=True)
            comp = paper_objective_components(final, initial, risk, final_conflicts, cfg)
            stats = origins(remaining, final_conflicts)
            risk0 = paper_objective_components(initial, initial, risk, conflicts, cfg)['ORISK']
            row = dict(test_id=test_id, optimizer_seed=seed, method=method, shared_stage1_hash=digest,
                initial_conflicts=len(conflicts), stage1_conflicts=len(remaining), final_conflicts=len(final_conflicts),
                final_conflict_pairs=count_conflict_pairs(final_conflicts),
                conflict_reduction_ratio=(len(remaining)-len(final_conflicts))/max(1,len(remaining)),
                resolved_conflicts=len(remaining)-len(final_conflicts),
                changed_flights=sum(_physically_changed(a,b) for a,b in zip(initial,final)),
                delayed_flights=comp['n_delay'], runtime_sec=runtime,
                final_fitness=paper_fitness(comp,reference,cfg,200,200),
                risk_increase_percent=100*(comp['ORISK']-risk0)/risk0 if risk0 else 0,
                new_conflict_pairs_created=stats['new_conflict_pairs'], **stats)
            write_json(directory / 'metrics.json', row)
            write_conflicts_csv(final_conflicts, directory / 'final_conflicts.csv')
            (directory / 'final_plans.pkl').write_bytes(pickle.dumps(final))
            pd.DataFrame(trace).to_csv(directory / 'convergence.csv', index=False)
            assert hashlib.sha256(pickle.dumps(base)+pickle.dumps(remaining)).hexdigest() == digest
            rows.append(row)
            pd.DataFrame(rows).to_csv(out / 'stage2_ab_comparison.csv', index=False)
            print(f'seed={seed} {method}: Nc={len(final_conflicts)} runtime={runtime:.2f}s', flush=True)
    frame = pd.DataFrame(rows)
    summary = frame.groupby('method').agg(final_conflicts_mean=('final_conflicts','mean'),
        final_conflicts_std=('final_conflicts','std'), zero_conflict_success_rate=('final_conflicts',lambda s: float((s==0).mean())),
        runtime_mean=('runtime_sec','mean'), changed_flights_mean=('changed_flights','mean'),
        delayed_flights_mean=('delayed_flights','mean'))
    summary.to_csv(out / 'stage2_ab_summary.csv')
    report = ['# Stage2 A/B Diagnostic Report', '', f'Test: {test_id}', '', summary.to_string(), '',
        'Paired seeds 0,1,2; one Stage1 per seed; identical serialized Stage2 inputs. Initial conflicts: 53 deterministic / 107 uncertain.',
        'Case B is legacy_targeted_stage2, NOT an exact paper/reference reproduction. Formal solvers unchanged.',
        'Legacy local reroute calls are adapted only at the experiment boundary to the CURRENT A* weights .8/.2, meter units and no vertical preference; old weight inflation and grid units are not imported.',
        'Runtime measures Stage2 solver only. Changed/delayed flights and risk compare against initial plans. Reduction ratio compares against shared Stage1.',
        'Exact survivors match unordered flight pair, cell and aligned waypoint indices, ignoring arrival times. Moved same pair counts remaining nonexact events on original pairs. New pairs counts distinct final pairs absent from Stage1; new points counts their events. Resolved original events = original count minus exact survivors, including moved events.',
        'New-pair creation counts final survivors, not all transient candidate conflicts. Rollback permits point increases when pair count decreases; it does not forbid replacement with new pairs.', '',
        frame[['optimizer_seed','method','final_conflicts','exact_survivors','moved_same_pair','new_conflict_points','new_conflict_pairs']].to_string(index=False), '',
        'A/B alone tests the combined grouping + targeted candidates + global rollback mechanism. Individual contributions require ablations. Three paired seeds support descriptive comparisons, not claims of statistical significance.',
        'If B reaches zero, propose B0 full / B1 no grouping / B2 no targeted candidates / B3 no rollback next; do not run them automatically or alter paper_strict.',
        'If B does not reach zero, scene structure, shared Stage1 output, 30-second threshold and local reroute feasibility remain hypotheses. Failure of these finite searches does NOT prove infeasibility.']
    for method, values in frame.groupby('method'):
        report.append(f"{method}: final conflicts mean={values.final_conflicts.mean():.3f}, sample std={values.final_conflicts.std():.3f}, "
                      f"zero successes={int((values.final_conflicts==0).sum())}/3; "
                      f"new final pairs mean={values.new_conflict_pairs.mean():.3f}.")
    (out / 'stage2_ab_report.md').write_text('\n\n'.join(report)+'\n', encoding='utf-8')
    manifest['status']='completed'
    write_json(out / 'manifest.json',manifest)
    print(out, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
