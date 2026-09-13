"""Ten paired GCEP diagnostics; all baseline and prior results stay untouched."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'1')

from datetime import datetime
from functools import partial
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from run_stage2_ab import origins,write_json
from scan_traffic_seeds import prepare_environment,validate_baseline
from src import fata
from src.adm_matching import adm_fata_optimize
from src.conflict_elite import run_diagnostic_fata
from src.conflict_detection import detect_conflicts,count_conflict_pairs,write_conflicts_csv
from src.paper_optimization import PaperReference,build_stage2_decision_layout,paper_objective_components,paper_fitness
from src.paper_scheduler import _physically_changed
from src.run_archive import git_metadata


def main():
    root=Path(__file__).resolve().parent
    branch=subprocess.check_output(['git','-c',f'safe.directory={root.parent.as_posix()}',
        'branch','--show-current'],cwd=root.parent,text=True).strip()
    if branch!='scheduling':raise RuntimeError('Requires scheduling branch')
    source=root/'outputs/a_plus_global_safeguard/20260913_154602_826781_A_plus'
    previous=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if previous['status']!='completed':raise RuntimeError('Prior A+ run incomplete')
    cfg=previous['config'];validate_baseline(cfg)
    grid,risk,hashes=prepare_environment(cfg,2025)
    if hashes!=previous['environment_hashes']:raise RuntimeError('Environment hash mismatch')
    ablation=Path(previous['stage1_source'])
    original=Path(json.loads((ablation/'manifest.json').read_text())['source'])
    initial=pickle.loads((original/'shared/initial_plans.pkl').read_bytes())
    ic=detect_conflicts(initial,cfg,uncertain=True)
    if (len(detect_conflicts(initial,cfg,uncertain=False)),len(ic))!=(53,107):raise RuntimeError('Initial scene mismatch')
    reference=PaperReference.from_initial(initial,risk,ic,cfg['optimization']['t_delay_max'])
    risk0=paper_objective_components(initial,initial,risk,ic,cfg)['ORISK']
    run_id=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_A_plus_elite'
    out=root/'outputs/a_plus_conflict_elite'/run_id;out.mkdir(parents=True,exist_ok=False)
    commit,dirty=git_metadata(root.parent)
    manifest=dict(run_id=run_id,status='validating',branch=branch,git_commit=commit,git_dirty=dirty,
        config=cfg,environment_hashes=hashes,shared_source=str(source),stage1_runs=0,
        mode='A_plus_conflict_elite',elite_count=1,
        injection_phase='After coherent current-population evaluation/incumbent/ADM; before next normal MLF/LPS proposal. Injected members mutate and resample context normally.',
        candidate_acceptance='Unchanged; no trial rollback or accepted mask',
        implementation_hash=hashlib.sha256((root/'src/conflict_elite.py').read_bytes()).hexdigest())
    write_json(out/'manifest.json',manifest)
    prior=pd.read_csv(source/'a_plus_raw_runs.csv');shared={}
    def csv(frame,path):
        frame=frame.copy();frame.insert(0,'run_id',run_id);frame.to_csv(path,index=False)
    for seed in range(10):
        directory=source/f'seed_{seed}/shared';blob=(directory/'shared_stage1_plans.pkl').read_bytes()
        old=detect_conflicts(pickle.loads(blob),cfg,uncertain=True)
        digest=hashlib.sha256(blob+pickle.dumps(old)).hexdigest()
        if not prior[prior.optimizer_seed==seed].stage1_input_hash.eq(digest).all():raise RuntimeError('Shared hash mismatch')
        shared[seed]=(blob,pickle.dumps(old),digest)
        dest=out/f'seed_{seed}';dest.mkdir()
        write_json(dest/'shared_stage1_reference.json',dict(run_id=run_id,optimizer_seed=seed,
            stage1_input_hash=digest,stage1_conflicts=len(old),source=str(directory)))
    csv(prior[prior.method=='A_plus_global_safeguard'].drop(columns='stage1_input_hash'),out/'strict_rollback_reference.csv')
    rows=[]
    manifest['status']='running';write_json(out/'manifest.json',manifest)
    try:
        for seed in range(10):
            blob,cblob,digest=shared[seed]
            for enabled in (False,True):
                base,old=pickle.loads(blob),pickle.loads(cblob)
                layout=build_stage2_decision_layout(base,old,cfg,grid,initial)
                records={};start=time.perf_counter()
                adapter=partial(run_diagnostic_fata,enable_elite=enabled,records=records)
                # Diagnostic hook is scoped; original FATA file and defaults unchanged.
                with patch.object(fata,'fata_optimize_paper',adapter):
                    result=adm_fata_optimize(base,initial,old,layout,cfg,grid,risk,reference,seed=seed)
                runtime=time.perf_counter()-start
                dest=out/f'seed_{seed}'/('A_plus' if enabled else 'A_current');dest.mkdir()
                stats=pd.DataFrame(records['stats']);stats.insert(0,'optimizer_seed',seed)
                csv(stats,dest/'population_generation_stats.csv')
                history=pd.DataFrame(records['history']);history['elite_risk_increase']=100*(history.elite_risk_increase-risk0)/risk0
                history.insert(0,'optimizer_seed',seed);csv(history,dest/'conflict_elite_history.csv')
                first=records['first_zero']
                diagnostic=dict(run_id=run_id,optimizer_seed=seed,ever_zero_conflict=records['ever_zero'],
                    first_zero_generation=first['generation'] if first else None,
                    first_zero_fitness=first['fitness'] if first else None,
                    first_zero_risk=100*(first['components']['ORISK']-risk0)/risk0 if first else None,
                    first_zero_changed_flights=first['changed_flights'] if first else None,
                    first_zero_delayed_flights=first['components']['n_delay'] if first else None)
                write_json(dest/'zero_discovery.json',diagnostic)
                if first:(dest/'first_zero_candidate.pkl').write_bytes(pickle.dumps(first))
                archive=records['archive']
                outputs=[('A_plus_paper_incumbent' if enabled else 'A_current_stage2',list(result.best_plans),
                    result.best_decision_vector,result.best_strategies,'paper_incumbent' if enabled else 'final')]
                if enabled:outputs.append(('A_plus_conflict_elite',archive.plans,archive.position,archive.context,'conflict_elite'))
                for method,plans,position,context,prefix in outputs:
                    fc=detect_conflicts(plans,cfg,uncertain=True);comp=paper_objective_components(plans,initial,risk,fc,cfg)
                    row=dict(run_id=run_id,optimizer_seed=seed,method=method,stage1_input_hash=digest,
                        stage1_conflicts=len(old),final_conflicts=len(fc),final_conflict_pairs=count_conflict_pairs(fc),
                        zero_conflict=len(fc)==0,ever_zero_conflict=records['ever_zero'],
                        first_zero_generation=first['generation'] if first else None,**origins(old,fc),
                        changed_flights=sum(_physically_changed(a,b) for a,b in zip(initial,plans)),
                        delayed_flights=comp['n_delay'],risk_increase_percent=100*(comp['ORISK']-risk0)/risk0,
                        final_paper_fitness=paper_fitness(comp,reference,cfg,200,200),runtime_sec=runtime,
                        elite_reinjection_count=records['reinjections'],
                        zero_found_then_lost_without_elite=bool(not enabled and records['ever_zero'] and len(fc)>0))
                    write_json(dest/(prefix+'_metrics.json' if enabled else 'metrics.json'),row)
                    (dest/(prefix+'_plans.pkl')).write_bytes(pickle.dumps(plans))
                    write_conflicts_csv(fc,dest/(prefix+'_conflicts.csv'))
                    conflict_csv=dest/(prefix+'_conflicts.csv');data=pd.read_csv(conflict_csv);csv(data,conflict_csv)
                    np.save(dest/(prefix+'_position.npy'),position);np.save(dest/(prefix+'_context.npy'),context)
                    rows.append(row)
                    print(f'seed={seed} {method}: points={len(fc)}, pairs={count_conflict_pairs(fc)}, risk={row["risk_increase_percent"]:.3f}%, everzero={records["ever_zero"]}',flush=True)
                    if not enabled:
                        saved=prior[(prior.optimizer_seed==seed)&(prior.method=='A_current_stage2')].iloc[0]
                        for name in ('final_conflicts','final_conflict_pairs','changed_flights','delayed_flights'):
                            if row[name]!=saved[name]:raise RuntimeError(f'A regression seed={seed} field={name}')
                        if not np.isclose(row['final_paper_fitness'],saved.final_fitness,rtol=1e-12):raise RuntimeError('A fitness regression')
                        np.testing.assert_array_equal(position,np.load(source/f'seed_{seed}/A_current/best_position.npy'))
                        np.testing.assert_array_equal(context,np.load(source/f'seed_{seed}/A_current/best_context.npy'))
                assert hashlib.sha256(pickle.dumps(base)+pickle.dumps(old)).hexdigest()==digest
                pd.DataFrame(rows).to_csv(out/'a_plus_elite_raw_runs.csv',index=False)
    except BaseException as exc:
        manifest.update(status='failed',error=repr(exc));write_json(out/'manifest.json',manifest);raise
    raw=pd.DataFrame(rows)
    summary=raw.groupby('method').agg(n_runs=('final_conflicts','size'),final_conflicts_mean=('final_conflicts','mean'),
        final_conflicts_std=('final_conflicts','std'),final_conflicts_median=('final_conflicts','median'),
        zero_conflict_success_rate=('zero_conflict','mean'),ever_zero_conflict_rate=('ever_zero_conflict','mean'),
        first_zero_generation_mean=('first_zero_generation','mean'),new_conflict_pairs_mean=('new_conflict_pairs','mean'),
        exact_survivors_mean=('exact_survivors','mean'),risk_increase_mean=('risk_increase_percent','mean'),
        risk_increase_std=('risk_increase_percent','std'),changed_flights_mean=('changed_flights','mean'),
        delayed_flights_mean=('delayed_flights','mean'),fitness_mean=('final_paper_fitness','mean'),
        runtime_mean=('runtime_sec','mean'),elite_reinjection_mean=('elite_reinjection_count','mean'))
    csv(summary.reset_index(),out/'a_plus_elite_summary.csv')
    report=['# Global Conflict Elite Preservation Report',f'Run ID: {run_id}',
        'Global Conflict Elite Preservation (GCEP) is an additional search-memory enhancement introduced on top of the reproduced ADM-FATA framework.',
        '```text\n'+summary.to_string()+'\n```',
        'Historical strict rollback reference: mean conflicts=10.9, zero rate=0/10, risk=0.755%; read only, not rerun.',
        'Candidate acceptance and ADM learning are unchanged. One archived elite is injected AFTER normal ADM learning into the coherent evaluated population, replacing lexicographic worst only if position AND context are absent. The next MLF/LPS proposal may mutate every row, including this elite; the next normal context sampling also remains unchanged. Archive copy is independent and repriced with current delta without historical decode/detection.',
        'Population statistics and ever-zero checks use the normal PRE-injection evaluated candidates. Elites are not repeated in same-generation ADM learning. Means exclude geometrically infeasible candidates. Lexicographic preference is NOT an extra paper-fitness penalty and need not monotonically decrease points when pairs decrease.',
        'Both paper-fitness incumbent and conflict archive are saved; no official output choice or objective change made.',
        'Baseline diagnostics reproduced the prior A positions, contexts, final counts and fitness for all ten seeds. Stage1 runs=0. No strict rollback, targeted candidates, grouping, legacy solver, relaxed safeguard or parameter changes.',
        'If no normal candidate reached zero, search discovery remains the evidence-supported limitation within this finite budget. If zero occurred but output is nonzero, final selection/preservation requires scrutiny; this does not prove an objective-scaling defect without further analysis.',
        'Archive monotonicity is a structural memory guarantee, not proof of overall convergence stability or statistical significance. Runtime is shared by both A+ output rows; never sum them as independent runs. No next-round enhancement implemented.']
    report_text='\n\n'.join(report)+'\n'
    (out/'a_plus_conflict_elite_report.md').write_text('\n'.join(line.rstrip() for line in report_text.split('\n')),encoding='utf-8')
    manifest['status']='completed';write_json(out/'manifest.json',manifest)
    print(out,flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
