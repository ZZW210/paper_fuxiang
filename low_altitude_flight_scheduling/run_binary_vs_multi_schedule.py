"""Paired paper_strict schedules differing only in binary vs multi-degree CI Top10."""
from __future__ import annotations
import argparse, copy, hashlib, json, pickle, time
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from src.config import load_config, resolve_scene_seeds
from src.conflict_detection import detect_conflicts, count_conflict_pairs, write_conflicts_csv
from src.conflict_network import build_conflict_network, collective_influence
from src.flight_plan import plan_paper_random_traffic, sample_paper_random_traffic
from src.grid import AirspaceGrid
from src.paper_scheduler import optimize_paper_schedule
from src.risk_map import generate_risk_map
from src.scene_diagnostics import coverage
from src.utils import set_random_seed

EXPECTED_BINARY=[98,47,63,49,56,25,72,64,1,4]
EXPECTED_MULTI=[73,47,49,56,33,78,42,25,1,63]

def _hash(value): return hashlib.sha256(pickle.dumps(value, protocol=5)).hexdigest()
def _pair(c): return tuple(sorted((c.plan_a,c.plan_b)))
def _ci(topology, degree):
    result={}
    for n in topology:
        shell=[v for v,d in nx.single_source_shortest_path_length(topology,n,cutoff=2).items() if d==2]
        result[n]=float((degree[n]-1)*sum(max(0,degree[v]-1) for v in shell))
    return result
def _keys(plans, conflicts, method):
    graph=build_conflict_network(plans,conflicts); degree=dict(graph.degree()); multi={p.id:0 for p in plans}
    for c in conflicts: multi[c.plan_a]+=1; multi[c.plan_b]+=1
    score=_ci(graph, degree if method=="binary" else multi)
    ids=sorted(graph, key=lambda n:(-score[n],n))[:10]; involved={p.id:0 for p in plans}
    for c in conflicts: involved[c.plan_a]+=1; involved[c.plan_b]+=1
    cov=coverage(conflicts,ids)
    table=pd.DataFrame([dict(rank=i+1,flight_id=n,CI_score=score[n],binary_degree=degree[n],multi_degree=multi[n],unique_conflict_partners=degree[n],conflict_point_involvement=involved[n],key_conflict_point_coverage=cov[0],key_conflict_pair_coverage=cov[1]) for i,n in enumerate(ids)])
    return ids,table
def _risk(plans,risk): return sum(float(risk[c]) for p in plans for c in p.path)
def _relation(original,current):
    old={(c.plan_a,c.plan_b,c.cell,c.idx_a,c.idx_b) for c in original}; now={(c.plan_a,c.plan_b,c.cell,c.idx_a,c.idx_b) for c in current}
    pairs={_pair(c) for c in original}; new=now-old
    return len(now&old),sum((x[0],x[1]) in pairs for x in new),len(new),len({_pair(c) for c in current}-{_pair(c) for c in original})
def _changed(a,b): return abs(a.atd-b.atd)>1e-6 or a.path!=b.path or not np.allclose(a.speed_profile,b.speed_profile)
def _run(plans,conflicts,keys,cfg,grid,risk,seed):
    cfg=copy.deepcopy(cfg); cfg["flight"]["random_seed"]=seed; set_random_seed(seed); start=time.perf_counter()
    result=optimize_paper_schedule(copy.deepcopy(plans),conflicts,keys,cfg,grid,risk,progress=False)
    stage=result.stage1.conflicts; final=detect_conflicts(result.final.plans,cfg,uncertain=True); initial_risk=_risk(plans,risk)
    srel=_relation(conflicts,stage); frel=_relation(conflicts,final)
    row=dict(initial_conflicts=len(conflicts),stage1_conflicts=len(stage),stage1_pairs=count_conflict_pairs(stage),stage1_reduction_ratio=1-len(stage)/len(conflicts),stage1_new_conflict_points=srel[2],stage1_new_conflict_pairs=srel[3],stage1_exact_survivors=srel[0],stage1_moved_same_pair=srel[1],stage1_changed_flights=sum(_changed(a,b) for a,b in zip(plans,result.stage1.plans)),stage1_delayed_flights=result.stage1.components["n_delay"],stage1_risk_increase_percent=(_risk(result.stage1.plans,risk)-initial_risk)/initial_risk*100,stage1_fitness=result.stage1.fitness,stage1_runtime_sec=result.stage1_seconds,stage1_dimension=result.dimensions[0],stage2_input_conflict_points=len(stage),stage2_input_conflict_pairs=count_conflict_pairs(stage),final_conflicts=len(final),final_pairs=count_conflict_pairs(final),stage2_reduction_ratio=1-len(final)/max(1,len(stage)),zero_conflict=int(not final),final_new_conflict_points=frel[2],final_new_conflict_pairs=frel[3],final_exact_survivors=frel[0],final_moved_same_pair=frel[1],total_changed_flights=sum(_changed(a,b) for a,b in zip(plans,result.final.plans)),delayed_flights=result.final.components["n_delay"],risk_increase_percent=(_risk(result.final.plans,risk)-initial_risk)/initial_risk*100,final_fitness=result.final.fitness,stage2_runtime_sec=result.stage2_seconds,total_runtime_sec=time.perf_counter()-start,stage2_dimension=result.dimensions[1])
    return row,result,final
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--environment-seed",type=int,default=2025); ap.add_argument("--traffic-seed",type=int,default=316); ap.add_argument("--n-jobs",type=int,default=8); args=ap.parse_args(); root=Path(__file__).resolve().parent
    cfg=load_config(root/"config.yaml",{"optimization":{"scheduler_mode":"paper_strict","n_jobs":args.n_jobs}}); resolve_scene_seeds(cfg,environment_seed=args.environment_seed,traffic_seed=args.traffic_seed)
    grid=AirspaceGrid.from_config(cfg,seed=args.environment_seed); risk=generate_risk_map(grid,cfg); plans=plan_paper_random_traffic(grid,risk,cfg,sample_paper_random_traffic(grid,cfg,100,args.traffic_seed)); det=detect_conflicts(plans,cfg,False); conflicts=detect_conflicts(plans,cfg,True)
    if (len(det),len(conflicts))!=(53,107): raise RuntimeError(f"expected 53/107, got {len(det)}/{len(conflicts)}")
    all_keys={m:_keys(plans,conflicts,m) for m in ("binary","multi")}
    if all_keys["binary"][0]!=EXPECTED_BINARY or all_keys["multi"][0]!=EXPECTED_MULTI: raise RuntimeError(f"Top10 validation failed: {all_keys['binary'][0]} / {all_keys['multi'][0]}")
    run_id=datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]+"_env2025_traffic316"; out=root/"outputs"/"multi_degree_full_schedule"/run_id; out.mkdir(parents=True)
    scene_hash=_hash(grid.obstacles); plan_hash=_hash([(p.id,p.path,p.eta_times) for p in plans]); conflict_hash=_hash(conflicts); rows=[]
    for seed in range(3):
      for method in ("binary","multi"):
        keys,keydf=all_keys[method]; directory=out/f"seed_{seed}"/method; directory.mkdir(parents=True); keydf.to_csv(directory/"key_flights.csv",index=False)
        row,result,final=_run(plans,conflicts,keys,cfg,grid,risk,seed); row.update(optimizer_seed=seed,method=method,key_flights=json.dumps(keys),key_conflict_point_coverage=float(keydf.key_conflict_point_coverage.iloc[0]),key_conflict_pair_coverage=float(keydf.key_conflict_pair_coverage.iloc[0]),scene_hash=scene_hash,plan_hash=plan_hash,initial_conflict_hash=conflict_hash); rows.append(row)
        pd.DataFrame([x for x in result.convergence if x["stage"]=="stage1"]).to_csv(directory/"stage1_convergence.csv",index=False); pd.DataFrame([x for x in result.convergence if x["stage"]=="stage2"]).to_csv(directory/"stage2_convergence.csv",index=False); (directory/"stage1_metrics.json").write_text(json.dumps({k:v for k,v in row.items() if k.startswith("stage1")},indent=2),encoding="utf-8"); (directory/"stage2_metrics.json").write_text(json.dumps({k:v for k,v in row.items() if k.startswith("stage2") or k.startswith("final")},indent=2),encoding="utf-8"); write_conflicts_csv(final,directory/"final_conflicts.csv"); pd.DataFrame(rows).to_csv(out/"binary_vs_multi_raw_runs.csv",index=False); print(f"seed={seed} method={method} final={row['final_conflicts']}",flush=True)
    raw=pd.DataFrame(rows); summary=raw.groupby("method").agg(n_runs=("method","size"),key_conflict_point_coverage=("key_conflict_point_coverage","mean"),stage1_conflicts_mean=("stage1_conflicts","mean"),stage1_conflicts_std=("stage1_conflicts","std"),stage1_reduction_mean=("stage1_reduction_ratio","mean"),final_conflicts_mean=("final_conflicts","mean"),final_conflicts_std=("final_conflicts","std"),zero_conflict_success_rate=("zero_conflict","mean"),changed_flights_mean=("total_changed_flights","mean"),delayed_flights_mean=("delayed_flights","mean"),risk_increase_mean=("risk_increase_percent","mean"),risk_increase_std=("risk_increase_percent","std"),final_fitness_mean=("final_fitness","mean"),stage1_runtime_mean=("stage1_runtime_sec","mean"),stage2_runtime_mean=("stage2_runtime_sec","mean"),total_runtime_mean=("total_runtime_sec","mean")).reset_index(); summary.to_csv(out/"binary_vs_multi_summary.csv",index=False)
    fig,ax=plt.subplots(figsize=(8,5));
    for method,g in raw.groupby("method"):
      for _,r in g.iterrows(): ax.plot([0,1,2],[107,r.stage1_conflicts,r.final_conflicts],alpha=.25,label=None)
      s=summary[summary.method.eq(method)].iloc[0]; ax.plot([0,1,2],[107,s.stage1_conflicts_mean,s.final_conflicts_mean],lw=3,label=method)
    ax.set(xticks=[0,1,2],xticklabels=["Initial","Stage1","Final"],ylabel="Conflict points",title="Binary vs multi-degree paired flow"); ax.legend(); fig.tight_layout(); fig.savefig(out/"binary_vs_multi_stage_flow.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5));
    for method,g in raw.groupby("method"): ax.scatter(g.final_conflicts,g.risk_increase_percent,label=method,s=55)
    ax.set(xlabel="Final conflicts",ylabel="Risk increase (%)",title="Risk versus final conflicts"); ax.legend(); fig.tight_layout(); fig.savefig(out/"binary_vs_multi_risk_conflict.png",dpi=180); plt.close(fig)
    delta=summary.loc[summary.method.eq("binary"),"stage1_conflicts_mean"].iloc[0]-summary.loc[summary.method.eq("multi"),"stage1_conflicts_mean"].iloc[0]; verdict="key-flight identification is supported" if delta>5 else "both key-flight identification and Stage2 remain bottlenecks" if delta>1 else "Stage2 remains the main bottleneck"
    lines=["# Binary vs Multi-degree Full Schedule", "", f"Binary Top10: {all_keys['binary'][0]}", f"Multi Top10: {all_keys['multi'][0]}", f"Overlap: {len(set(all_keys['binary'][0])&set(all_keys['multi'][0]))}.", "", "## Summary"]
    for record in summary.to_dict("records"):
      lines.append("- " + ", ".join(f"{key}={value}" for key,value in record.items()))
    lines += ["", f"Delta_stage1={delta:.3f}. Conclusion: {verdict}.", "Risk and changed/delayed-flight metrics are in the summary; no parameter other than key-flight ranking changed."]
    (out/"binary_vs_multi_full_schedule_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    (out/"manifest.json").write_text(json.dumps(dict(environment_seed=2025,traffic_seed=316,initial_deterministic_conflicts=53,initial_uncertain_conflicts=107,top_k=10,ci_l=2,seeds=[0,1,2],methods=["binary","multi"],scene_hash=scene_hash,plan_hash=plan_hash,initial_conflict_hash=conflict_hash,stages_executed=["Stage1","Stage2"]),indent=2),encoding="utf-8"); print(out)
if __name__=="__main__": main()
