"""Paired A / A_plus_global_safeguard study; never changes default scheduler."""
from __future__ import annotations

import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')

from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time

import numpy as np
import pandas as pd

from run_stage2_ab import origins,write_json
from scan_traffic_seeds import prepare_environment,validate_baseline
from src.adm_matching import adm_fata_optimize
from src.global_safeguard import adm_fata_optimize_with_safeguard
from src.conflict_detection import detect_conflicts,count_conflict_pairs,write_conflicts_csv
from src.paper_optimization import PaperReference,build_stage2_decision_layout,paper_objective_components,paper_fitness
from src.paper_scheduler import _physically_changed
from src.run_archive import git_metadata


def main():
    root = Path(__file__).resolve().parent
    branch = subprocess.check_output(['git','-c',f'safe.directory={root.parent.as_posix()}',
        'branch','--show-current'],cwd=root.parent,text=True).strip()
    if branch!='scheduling':
        raise RuntimeError('A+ paired experiment requires scheduling branch')
    ablation = root/'outputs/stage2_mechanism_ablation/20260913_115912_599871_ablation'
    metadata = json.loads((ablation/'manifest.json').read_text(encoding='utf-8'))
    if metadata['status']!='completed':
        raise RuntimeError('Ablation not completed')
    original = Path(metadata['source'])
    cfg = metadata['config']
    validate_baseline(cfg)
    if (cfg['fata']['NP'],cfg['fata']['Ngen_max_stage2'],cfg['fata']['Parf'])!=(50,200,.2):
        raise RuntimeError('Budget mismatch')
    grid,risk,hashes = prepare_environment(cfg,2025)
    if hashes!=metadata['environment_hashes']:
        raise RuntimeError('Environment mismatch')
    initial = pickle.loads((original/'shared/initial_plans.pkl').read_bytes())
    uncertain = detect_conflicts(initial,cfg,uncertain=True)
    if (len(detect_conflicts(initial,cfg,uncertain=False)),len(uncertain))!=(53,107):
        raise RuntimeError('Initial conflict mismatch')
    reference = PaperReference.from_initial(initial,risk,uncertain,cfg['optimization']['t_delay_max'])
    risk0 = paper_objective_components(initial,initial,risk,uncertain,cfg)['ORISK']
    test_id = datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_A_plus'
    out = root/'outputs/a_plus_global_safeguard'/test_id
    out.mkdir(parents=True,exist_ok=False)
    commit,dirty = git_metadata(root.parent)
    manifest = dict(status='validating',test_id=test_id,branch=branch,git_commit=commit,git_dirty=dirty,
        config=cfg,environment_hashes=hashes,stage1_runs=0,stage1_source=str(ablation),
        mode='A_plus_global_safeguard',attribution='Project enhancement, not original paper rollback',
        parent_fitness_policy='Cached components repriced using current-generation delta; no parent route/detection repeat',
        final_selection='Unchanged paper fitness incumbent among retained individuals; conflict lex elite diagnostic',
        adm_policy='Only accepted finite individuals teach; initialization finite individuals teach unchanged formula',
        implementation_hash=hashlib.sha256((root/'src/global_safeguard.py').read_bytes()).hexdigest())
    write_json(out/'manifest.json',manifest)
    shared = {}
    source_rows = pd.read_csv(ablation/'raw_runs.csv')
    for seed in range(10):
        directory = (original if seed<3 else ablation)/f'seed_{seed}'
        blob = (directory/'shared_stage1_plans.pkl').read_bytes()
        base = pickle.loads(blob)
        conflicts = detect_conflicts(base,cfg,uncertain=True)
        digest = hashlib.sha256(blob+pickle.dumps(conflicts)).hexdigest()
        if not source_rows[source_rows.optimizer_seed==seed].stage1_input_hash.eq(digest).all():
            raise RuntimeError(f'Seed {seed} shared input mismatch')
        saved = pd.read_csv(directory/'shared_stage1_conflicts.csv')
        dest = out/f'seed_{seed}/shared'
        dest.mkdir(parents=True)
        (dest/'shared_stage1_plans.pkl').write_bytes(blob)
        write_conflicts_csv(conflicts,dest/'shared_stage1_conflicts.csv')
        pd.testing.assert_frame_equal(saved,pd.read_csv(dest/'shared_stage1_conflicts.csv'))
        write_json(dest/'metrics.json',dict(optimizer_seed=seed,stage1_input_hash=digest,
            stage1_conflicts=len(conflicts),source=str(directory)))
        shared[seed]=(blob,pickle.dumps(conflicts),digest)
    manifest['status']='running';write_json(out/'manifest.json',manifest)
    rows,reason_rows = [],[]
    try:
        for seed in range(10):
            for method,dname in (('A_current_stage2','A_current'),('A_plus_global_safeguard','A_plus')):
                blob,cblob,digest = shared[seed]
                base,conflicts = pickle.loads(blob),pickle.loads(cblob)
                dest = out/f'seed_{seed}'/dname
                dest.mkdir()
                layout = build_stage2_decision_layout(base,conflicts,cfg,grid,initial)
                start = time.perf_counter()
                if dname=='A_current':
                    result = adm_fata_optimize(base,initial,conflicts,layout,cfg,grid,risk,reference,seed=seed)
                    diagnostic = []
                else:
                    def progress(row):
                        if row['generation']%20==0:
                            print(f'seed={seed} A+ gen={row["generation"]} Nc={row["best_conflict_points"]} acceptance={row["acceptance_rate"]:.3f}',flush=True)
                    result,diagnostic,masks,lex = adm_fata_optimize_with_safeguard(base,initial,conflicts,
                        layout,cfg,grid,risk,reference,seed=seed,callback=progress)
                    np.save(dest/'accepted_mask.npy',masks)
                    (dest/'conflict_lexicographic_elite.pkl').write_bytes(pickle.dumps(lex))
                    pd.DataFrame(diagnostic).to_csv(dest/'rollback_diagnostics.csv',index=False)
                    for d in diagnostic:
                        reason_rows.append(dict(optimizer_seed=seed,**d))
                runtime = time.perf_counter()-start
                final = list(result.best_plans)
                final_conflicts = detect_conflicts(final,cfg,uncertain=True)
                comp = paper_objective_components(final,initial,risk,final_conflicts,cfg)
                accepted = sum(d['accepted_trials'] for d in diagnostic)
                rollback = sum(d['rolled_back_trials'] for d in diagnostic)
                row = dict(optimizer_seed=seed,method=method,stage1_input_hash=digest,
                    stage1_conflicts=len(conflicts),final_conflicts=len(final_conflicts),
                    final_conflict_pairs=count_conflict_pairs(final_conflicts),zero_conflict=len(final_conflicts)==0,
                    conflict_reduction_ratio=(len(conflicts)-len(final_conflicts))/max(1,len(conflicts)),
                    **origins(conflicts,final_conflicts),
                    changed_flights=sum(_physically_changed(a,b) for a,b in zip(initial,final)),
                    delayed_flights=comp['n_delay'],risk_increase_percent=100*(comp['ORISK']-risk0)/risk0,
                    final_fitness=paper_fitness(comp,reference,cfg,200,200),runtime_sec=runtime,
                    rollback_count=rollback,accepted_trial_count=accepted,
                    acceptance_rate=accepted/(accepted+rollback) if accepted+rollback else float('nan'))
                assert np.isclose(row['final_fitness'],result.best_fitness,rtol=1e-12)
                assert hashlib.sha256(pickle.dumps(base)+pickle.dumps(conflicts)).hexdigest()==digest
                write_json(dest/'metrics.json',row)
                write_conflicts_csv(final_conflicts,dest/'final_conflicts.csv')
                (dest/'final_plans.pkl').write_bytes(pickle.dumps(final))
                np.save(dest/'probability_history.npy',np.asarray(result.probability_history))
                np.save(dest/'best_position.npy',result.best_decision_vector)
                np.save(dest/'best_context.npy',result.best_strategies)
                pd.DataFrame(dict(generation=np.arange(1,201),fitness=result.convergence)).to_csv(dest/'convergence.csv',index=False)
                rows.append(row)
                pd.DataFrame(rows).to_csv(out/'a_plus_raw_runs.csv',index=False)
                print(f'seed={seed} {method}: final={len(final_conflicts)}, risk={row["risk_increase_percent"]:.3f}%, runtime={runtime:.2f}s',flush=True)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));write_json(out/'manifest.json',manifest);raise
    frame = pd.DataFrame(rows)
    summary = frame.groupby('method').agg(n_runs=('final_conflicts','size'),final_conflicts_mean=('final_conflicts','mean'),
        final_conflicts_std=('final_conflicts','std'),final_conflicts_median=('final_conflicts','median'),
        zero_conflict_success_rate=('zero_conflict','mean'),new_conflict_pairs_mean=('new_conflict_pairs','mean'),
        exact_survivors_mean=('exact_survivors','mean'),changed_flights_mean=('changed_flights','mean'),
        delayed_flights_mean=('delayed_flights','mean'),risk_increase_mean=('risk_increase_percent','mean'),
        risk_increase_std=('risk_increase_percent','std'),fitness_mean=('final_fitness','mean'),runtime_mean=('runtime_sec','mean'))
    summary.to_csv(out/'a_plus_summary.csv')
    reasons = pd.DataFrame(reason_rows)
    fields = ['rollback_due_to_pair_increase','rollback_due_to_point_increase','rollback_due_to_fitness_worse',
        'rollback_due_to_infeasible_route','new_pair_trials_rejected','rejected_lower_risk_equal_conflicts']
    reason_summary = reasons.groupby('optimizer_seed')[fields].sum()
    reason_summary.to_csv(out/'rollback_reason_summary.csv')
    frame[['optimizer_seed','method','final_conflicts','risk_increase_percent','delayed_flights','changed_flights']].to_csv(out/'risk_conflict_comparison.csv',index=False)
    a,plus = summary.loc['A_current_stage2'],summary.loc['A_plus_global_safeguard']
    report = ['# A vs A+ Global Safeguard Report','Project enhancement: A_plus_global_safeguard. paper_strict remains the unmodified reproduction baseline.',
        '```text\n'+summary.to_string()+'\n```',
        f'A mean conflicts={a.final_conflicts_mean:.3f}; A+={plus.final_conflicts_mean:.3f}. Zero rates: {a.zero_conflict_success_rate:.1%} vs {plus.zero_conflict_success_rate:.1%}.',
        f'New final pair means: {a.new_conflict_pairs_mean:.3f} vs {plus.new_conflict_pairs_mean:.3f}; exact survivors: {a.exact_survivors_mean:.3f} vs {plus.exact_survivors_mean:.3f}.',
        'Rollback totals:\n```text\n'+reason_summary.sum().to_string()+'\n```',
        f'Risk means: A={a.risk_increase_mean:.3f}%, A+={plus.risk_increase_mean:.3f}%. Delayed flights: {a.delayed_flights_mean:.3f} vs {plus.delayed_flights_mean:.3f}. Changes: {a.changed_flights_mean:.3f} vs {plus.changed_flights_mean:.3f}.',
        f'Ideal criterion (better zero rate and mean risk <=2%): {bool(plus.zero_conflict_success_rate>a.zero_conflict_success_rate and plus.risk_increase_mean<=2)}.',
        'Acceptance is strictly candidate-level (pairs, points, current-generation paper fitness). Pair decrease can accept point or risk increase; the rule does not prohibit all novel pairs. Equal metrics retain parent. Infeasible routes never pass a finite parent.',
        'Parent position/context/components/conflicts are inherited on rollback. Parent and elites fitness are recomputed from cached components under current delta without decoding/detection. Trial evaluation uses existing globally equivalent incremental detector and caches. Final output is the original fitness incumbent among retained candidates, not the diagnostic lexicographic elite.',
        'FATA good-point initialization, MLF/LPS formula, sequential proposal draws, ADM formula and learning rate unchanged. Accepted-mask filtering is the only ADM learning eligibility addition. Generation 1 initializes parents and does not count toward trial acceptance rates.',
        'All 10 Stage1 inputs reused; seed 0..2 are stored in the original A/B directory and referenced by ablation provenance. No scene regeneration, Stage1 run, B rerun, legacy candidates or grouping. Environment arrays regenerated only to verify identical hashes.',
        'Ten paired seeds are descriptive evidence, not proof of global optimality or statistical significance. No relaxed safeguard, altered objective/weights or default-mode changes. Previous legacy B average risk was 10.826% in the ten-seed ablation, cited only as diagnostic reference.']
    if plus.risk_increase_mean>5 or frame[frame.method=='A_plus_global_safeguard'].risk_increase_percent.max()>5:
        text = 'Strict safeguard may block transitions with temporarily worse conflicts, even if risk decreases. Equal conflict counts already permit lower PAPER FITNESS, not risk alone.\n\n'+reason_summary.sum().to_string()+'\n\nThese counts do not establish causality. A relaxed safeguard requires a separate paired experiment; none implemented.\n'
        (out/'risk_vs_conflict_tradeoff.md').write_text(text,encoding='utf-8')
    (out/'a_plus_global_safeguard_report.md').write_text('\n\n'.join(report)+'\n',encoding='utf-8')
    manifest['status']='completed';write_json(out/'manifest.json',manifest)
    print(out,flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
