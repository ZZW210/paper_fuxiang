"""Strict scheduling consistency checks and scientific reports."""
from __future__ import annotations

import ast
import base64
import copy
import html
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

from . import fata, paper_optimization as model, paper_scheduler, adm_matching
from .conflict_detection import detect_conflicts
from .conflict_network import select_paper_key_flights
from .flight_plan import FlightPlan, compute_eta_times
from .grid import AirspaceGrid


REQUIREMENTS = [
    ("两阶段、重新探测", "目标论文3.2.1", "paper_scheduler.optimize_paper_schedule", "一致", "无"),
    ("普通CI前10%", "式(29)、表4及4.3", "conflict_network.select_paper_key_flights", "本轮未修改", "同分ID排序；初始网络是既有数据近似"),
    ("允许单独及组合策略", "目标论文3.1.1、3.2.1优先于参考[6]", "paper_optimization._decode", "恢复目标要求", "参考[6]单UAV单策略仅为该参考自身假设，已停用"),
    ("Stage1七种非空组合", "用户activation编码要求", "paper_optimization.decode_activation_genes", "编码假设", "三gene [0,1]，>=.5激活；均未激活则取最大gene，同分ID，保证非空；不按冲突类型指定"),
    ("ATD连续[1,3600]，提前/延后<=1800", "式(38)-(40)、表3", "paper_optimization.paper_atd_bounds", "范围一致", "Stage2范围仍相对original ETD；未匹配schedule则保留Stage1 ATD，匹配后优化新的绝对ATD"),
    ("连续逐航段速度[5,20]", "v_ijk、式(41)、表3", "paper_optimization._build_layout", "范围一致；局部窗口假设", "Stage1完整profile；Stage2仅操作SPEED标签关联航段，其余Stage1速度不变；索引窗口半径4"),
    ("xyz改航、26邻域、端点、无障碍", "式(35)、(37)、(42)", "paper_optimization._via_route", "via-cell+A*近似", "变长路径编码未公开；全范围xyz取整，内部0-based；via等于原窗口中点时精确保留原路径no-op"),
    ("两端都开放，允许不改航", "本轮用户要求", "paper_optimization.PaperFlightBlock", "no-op编码假设", "Stage2每UAV局部窗口有[0,1]改航开关，由FATA决定；无owner或奇偶actor；no-op不等于实际改航"),
    ("组合下保留已有速度", "目标组合要求；映射细节未知", "paper_optimization._via_route", "弧长映射假设", "相同路径精确保留profile；替换路径按归一化弧长投影当前速度，没有新速度决策"),
    ("Stage2在Stage1基础上增量优化", "目标论文3.2.1", "paper_optimization._decode", "已修复working base", "布局使用Stage1路径及冲突索引；成本和changed仍比较initial，不回原计划重做"),
    ("同flight累积多个冲突策略", "目标组合要求，ADM本次扩展", "paper_optimization.stage2_flight_strategies", "共享变量编码假设", "ATD每flight共享；重叠速度段共享；重叠空间窗口共享via-cell；仅解码合并变量块，不合并或改写species"),
    ("Tdelay绝对值、Tair、ORISK、Nc points", "式(31)-(33)、(51)", "paper_optimization.paper_objective_components", "结构一致", "初始生成、风险地图及冲突探测本轮未修改；ORISK是已有栅格风险而非真实概率场复原"),
    ("默认raw Eq.(51)", "打印式(51)及本轮要求", "paper_optimization.paper_fitness", "直算打印公式，不能声称复原作者尺度", "论文说明无量纲化但变换未知；旧约[0,1]尺度使100/1000罚项支配；不反向调系数"),
    ("initial-reference仅实验", "无量纲化未知；本轮对照要求", "paper_optimization.normalize_paper_objectives", "initial_reference_experimental假设", "两阶段共用initial固定参考；不得跨尺度比较fitness大小或自动选更好模式"),
    ("动态delta、gamma=5", "式(49)", "paper_optimization.conflict_weight_delta", "一致", "每代用当前delta重评估保存best"),
    ("Sfit打印罚项及固定权重", "式(50)-(51)、表5", "paper_optimization.paper_fitness", "两种尺度均保持罚项", "wc=.8，wd=.25，wt=.25，wr=.5；nbattery=Tair>1200；ndelay=ATD>ETD+1e-6"),
    ("无3%cap、changed cap、pair目标", "模型约束与实验结果区分", "paper_optimization._decode", "一致", "无百万冲突罚项、工程repair、单步rollback或结果强制比例"),
    ("匹配每个剩余冲突点", "目标3.2.1、参考[6]3.2.1", "adm_matching.adm_fata_optimize", "P形状(m,3)", "仅路径窗口共享，不合并概率矩阵或采样标签"),
    ("均匀初始化、独立采样、并发评价", "参考[6]Algorithm2及用户等概率要求", "adm_matching.sample_strategy_species", "保留原始sampled species", "1/3为用户给定；参考可用Algorithm2未明确列出初始数值；无类型先验"),
    ("原始优势种群频率更新，lrate=.5", "参考[6]4.3及Algorithm2第18行", "adm_matching.update_probability_matrix", "公式一致", "无reconciliation或owner投票；dominantNo未知，用round(NP*.20)，无可行个体则不更新"),
    ("Stage2再次改进FATA", "目标3.2.1及3.2.2", "adm_matching.adm_fata_optimize", "组合实现仍为近似", "未公开替换ISFS后的源代码；每代采样、评价、更新P、MLF/LPS，不声称复刻双层ISFS"),
    ("佳点集和原始FATA，无Gaussian增补", "FATA.m、式(48)", "fata.fata_optimize_paper", "本轮未改FATA", "保留scalar-rand reset、IP/p、两相折射和全反射，无噪声、warm start或Gaussian local search"),
    ("几何不可行解处理", "路径约束，数值细节未知", "paper_optimization.InfeasiblePaperRoute", "数值假设", "不可行fitness=inf，有限哨兵仅用于MLF积分；不回滚原路径掩盖不可行"),
    ("normal 50/200/200，quick 20/50/50", "表5及用户预算", "paper_scheduler.optimize_paper_schedule", "保持结构及预算", "无剩余冲突时明确跳过Stage2；8进程只并行fitness，随机更新和ADM顺序固定"),
    ("历史、提前/延后、实际改航诊断", "本轮诊断要求", "paper_scheduler.write_paper_diagnostics", "实现", "策略计数允许重叠和no-op；实际改航另列；最终changed对initial，Stage2增量对Stage1"),
    ("定性现象只做事后比较", "本轮要求及论文实验现象", "paper_consistency.qualitative_observations", "不设硬约束", "未改善项如实列出；大幅提前且ndelay=0仅标记，不限制或补repair"),
    ("两种尺度分别运行，不自动选择", "本轮对照实验要求", "compare_objective_scales.write_scale_comparison", "独立实验目录", "校验相同初始输入、seed和预算，CSV保留真实结果，raw仍默认"),
]


def run_consistency_checks(cfg):
    checks = []
    def check(name, predicate):
        try:
            passed = bool(predicate())
        except Exception:
            passed = False
        checks.append((name, passed))
    opt, weights = cfg["optimization"], cfg["fata"]
    sources = [inspect.getsource(module) for module in (model, adm_matching, paper_scheduler)]
    ps, ads, ss = sources
    fs = inspect.getsource(fata.fata_optimize_paper)
    metrics = pd.DataFrame(dict(flight_id=range(100), collective_influence=range(100), weighted_collective_influence=range(100, 0, -1)))
    path = [(i, 2, 1) for i in range(1, 6)]
    plan = FlightPlan(0, path[0], path[-1], path, 1000, compute_eta_times(path, 1000, 10.0, (100,100,30)), [10.0]*4, 5.0, 40.0)
    grid = AirspaceGrid(shape=(8,8,4), obstacles=np.zeros((8,8,4), dtype=bool))
    risk = np.ones(grid.shape)
    layout = model.build_stage1_decision_layout([plan], [0], [], cfg, grid)
    block = layout.blocks[0]
    vector = (layout.lower + layout.upper) / 2
    vector[block.activation_genes] = [0,1,0]
    vector[block.speed_genes] = [10,11.374,10,10]
    decoded = model.decode_stage1_solution(vector,[plan],layout,cfg,grid,risk)[0]
    check("Stage1 fixed CI top 10%", lambda: opt["stage1_key_ratio"]==.1 and opt["stage1_key_selection_mode"]=="fixed_ci" and select_paper_key_flights(metrics,100)==list(range(99,89,-1)))
    combos = {model.decode_activation_genes([.8 if mask & (1<<i) else .2 for i in range(3)]) for mask in range(1,8)}
    check("Stage1 seven nonempty strategy combinations", lambda: len(combos)==7 and model.decode_activation_genes([.1,.3,.2])==(1,))
    check("ATD continuous and early departure allowed", lambda: model.paper_atd_bounds(plan,cfg)==(1,2800))
    check("per-segment continuous speed [5,20]", lambda: opt["speed_range"]==[5,20] and decoded.speed_profile[1]==11.374 and decoded.eta_times[:2]==plan.eta_times[:2])
    check("local reroute x/y/z optimization", lambda: bool(block.reroute_genes) and all(g.stop-g.start==3 for g in block.reroute_genes))
    check("paper Eq.(49) delta", lambda: model.conflict_weight_delta(200,200)==.9 and model.conflict_weight_delta(0,200)==1 and weights["gamma"]==5)
    c = dict(Tdelay=200,Tair=100,ORISK=5,Nc=2,n_delay=2,n_battery=1)
    ref = model.PaperReference(1800,100,5,2)
    raw_cfg = copy.deepcopy(cfg)
    raw_cfg["optimization"]["paper_objective_scale_mode"]="raw_equation"
    obj = .2*(.25*200+.25*100+.5*5)+.8*.9*2*100
    check("paper Eq.(50) penalties unchanged", lambda: np.isclose(model.paper_fitness(c,ref,raw_cfg,200,200),obj+1200))
    check("paper Eq.(51) raw_equation default", lambda: opt["paper_objective_scale_mode"]=="raw_equation" and np.isclose(model.paper_fitness({**c,"n_delay":0,"n_battery":0},ref,cfg,200,200),obj))
    exp = copy.deepcopy(cfg)
    exp["optimization"]["paper_objective_scale_mode"]="initial_reference_experimental"
    check("initial-reference experimental mode available", lambda: np.isclose(model.paper_fitness(c,ref,exp,200,200),1200+.2*(.25*200/1800+.25+.5)+.8*.9))
    check("no hard conflict penalty or delay/changed caps", lambda: all(s not in ps for s in ("conflict_hard_penalty","conflict_pair_penalty","delay_count_cap","max_changed_flight_ratio")))
    check("no handcrafted conflict-type priors", lambda: "head_to_head" not in ads and "conflict_type" not in ads)
    check("ADM uniformly initialized", lambda: np.array_equal(adm_matching.initialize_probability_matrix(3),np.full((3,3),1/3)))
    check("ADM learning rate = 0.5", lambda: cfg["adm"]["learning_rate"]==.5 and np.allclose(adm_matching.update_probability_matrix(np.full((1,3),1/3),np.array([[2]])),[[1/6,1/6,2/3]]))
    check("ADM updates original sampled species", lambda: "return sample_strategy_species(probability, size, rng)" in ads and "dominant_species = np.asarray(contexts)[ranked]" in ads)
    check("no single-UAV reconciliation or owners", lambda: "enforce_single_strategy_per_flight" not in ads and "owners" not in ads)
    other = plan.copy()
    other.id = 1
    current = [plan.copy(),other.copy()]
    current[0].atd=990
    current[0].speed_profile=[11.374]*4
    current[0].paper_strategies=(0,1)
    current[0].eta_times=compute_eta_times(path,990,current[0].speed_profile,grid.cell_size)
    conflicts=detect_conflicts(current,cfg)
    l2=model.build_stage2_decision_layout(current,conflicts,cfg,grid,[plan,other])
    v2=(l2.lower+l2.upper)/2
    for b in l2.blocks:
        v2[b.reroute_enable_genes]=0
    d2=model.decode_stage2_solution(v2,current,[plan,other],l2,np.full(len(conflicts),2),cfg,grid,risk)
    check("Stage2 preserves Stage1 ATD/speed and accumulates strategies", lambda: d2[0].atd==990 and d2[0].speed_profile==current[0].speed_profile and model.flight_strategies(d2[0])==(0,1,2))
    check("both endpoints have legal no-op variables", lambda: {b.flight_id for b in l2.blocks}=={0,1} and all(b.reroute_enable_genes for b in l2.blocks) and all(l2.lower[b.atd_gene]<=current[b.flight_id].atd<=l2.upper[b.atd_gene] for b in l2.blocks))
    check("Stage2 again runs improved FATA concurrently", lambda: "fata.fata_optimize_paper(" in ads and "adm_fata_optimize(" in ss and "objective_with_context=objective.context_fitness" in ads)
    forbidden={"greedy_time_deconfliction","repair_conflicts","try_apply_action_with_rollback","_delay_candidates","_speed_factors","independent_matching_deconfliction"}
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(ast.parse('\n'.join(sources))) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    check("no greedy rollback in strict mode", lambda: not calls.intersection(forbidden))
    seen=[]
    def objective(v,g):
        seen.append(v.copy())
        return float(np.sum(v*v))
    fata.fata_optimize_paper(objective,[0,0],[1,1],2,population=4,max_iter=1,n_jobs=1)
    check("good-point initialization has no noise", lambda: np.array_equal(np.asarray(seen),fata.good_point_set(4,2,np.zeros(2),np.ones(2))))
    check("no Gaussian local search", lambda: ".normal(" not in fs and "local_count" not in fs)
    for name,key,value in (("NP=50","NP",50),("Stage1 Ngen=200","Ngen_max_stage1",200),("Stage2 Ngen=200","Ngen_max_stage2",200)):
        check(name,lambda k=key,v=value: weights[k]==v)
    return checks


def qualitative_observations(s):
    initial,stage1,final=(s.get(k,0) for k in ("stage1_initial_conflict_points","stage1_final_conflict_points","stage2_final_conflict_points"))
    penalty = 1000 * s.get("final_n_battery", 0) + 100 * s.get("final_n_delay", 0)
    fitness = s.get("final_paper_fitness", 0)
    return [
        f"终代罚项={penalty:.3f}，占fitness {penalty/max(abs(fitness),np.finfo(float).eps):.1%}；"
        "changed架数及Stage2增量修改架数不直接入目标，不能保证少改动；不为拟合现象新增惩罚。",
        f"Stage1冲突点{initial} -> {stage1}，减少{(initial-stage1)/max(1,initial):.1%}；整体点数不等于作者连续冲突私有数据。",
        f"Stage1 schedule-only={s.get('stage1_schedule_only_count',0)}/{s.get('stage1_key_flight_count',0)}，组合flight={s.get('stage1_strategy_combination_count',0)}；若仍只用schedule，如实列出。",
        f"实际改航flight：Stage1={s.get('stage1_actual_rerouted_flight_count',0)}，Stage2相对Stage1={s.get('stage2_actual_rerouted_flight_count',0)}。",
        f"Stage2冲突点{stage1} -> {final}，继续下降={final<stage1}；working base为Stage1，不代表必然找到更小改动。",
        f"最终changed={s.get('changed_flight_count_two_stage',0)}，少于旧版61={s.get('changed_flight_count_two_stage',0)<61}；Stage2增量修改{s.get('stage2_modified_flight_count',0)}架，新增changed {s.get('stage2_newly_changed_flight_count',0)}架。",
        f"提前{s.get('advanced_flight_count',0)}架，延后{s.get('delayed_flight_count',0)}架；平均/最大绝对偏移{s.get('mean_absolute_atd_shift',0):.3f}/{s.get('max_absolute_atd_shift',0):.3f}s。",
        f"n_delay=0且Tdelay>20000观察标记={s.get('large_advance_without_delay_observed',False)}；提前也计入绝对Tdelay，数学上可出现，此标记仅诊断，不触发限制或repair。",
    ]


def stage1_coverage_observation(out: Path, s):
    conflicts = pd.read_csv(out / "conflicts_uncertain.csv")
    assignment = pd.read_csv(out / "paper_stage1_strategy_assignment.csv")
    keys = set(assignment["flight_id"])
    related = conflicts.plan_a.isin(keys) | conflicts.plan_b.isin(keys)
    untouched = int((~related).sum())
    return (f"固定CI关键计划涉及初始冲突点{int(related.sum())}/{len(conflicts)}；"
            f"其余{untouched}点两端都不参与Stage1，故Stage1整体Nc不能低于{untouched}。"
            f"本次Stage1 Nc={s.get('stage1_final_conflict_points')}；该覆盖诊断不改变CI或生成数据。")


def write_implementation_report(out: Path, s):
    lines=["# Paper-strict implementation report","","本轮恢复组合策略、Stage1增量基准、双端变量和原始ADM标签，默认raw_equation。初始生成、风险地图、冲突探测、复杂网络及CI实现均未修改。","",
           "| 原论文或用户要求 | 依据 | 对应代码函数 | 是否一致/近似 | 原因 |","|---|---|---|---|---|"]
    for requirement,evidence,code,status,reason in REQUIREMENTS:
        module=code.split('.')[0]
        target=f"../src/{module}.py" if module!="compare_objective_scales" else "../compare_objective_scales.py"
        lines.append(f"| {requirement} | {evidence} | [{code}]({target}) | {status} | {reason} |")
    lines += ["","## 本次运行","","真实参数与结果来自paper_run_config.json、paper_convergence.csv及metrics_summary.csv。",""]
    for k in ("paper_objective_scale_mode","seed","NP","Ngen_max_stage1","Ngen_max_stage2","n_jobs","stage1_initial_conflict_points","stage1_final_conflict_points","stage2_final_conflict_points","changed_flight_count_two_stage","stage1_strategy_combination_count","stage2_strategy_combination_count","final_Tdelay","final_Tair","final_ORISK","final_n_delay","final_n_battery","final_paper_fitness","stage1_fata_time","stage2_fata_time","total_runtime"):
        lines.append(f"- `{k}`: {s.get(k,'not recorded')}")
    lines += ["","## 定性现象观察","","事后报告，不达预期也保留；策略参与数可重叠或no-op，实际改航另列。",""]+['- '+v for v in qualitative_observations(s)]
    lines += ["","## OLD / NEW","","| 指标 | 旧版 | 本次运行 |","|---|---|---|"]
    comparison=[("冲突点","130 -> 98 -> 32",f"{s.get('stage1_initial_conflict_points')} -> {s.get('stage1_final_conflict_points')} -> {s.get('stage2_final_conflict_points')}"),("changed",61,s.get('changed_flight_count_two_stage')),("Tdelay/s",25188.77,s.get('final_Tdelay'))]
    for stage,old in ((1,"10/0/0"),(2,"23/27/1")):
        comparison.append((f"Stage{stage} schedule/speed/reroute",old,'/'.join(str(s.get(f'stage{stage}_strategy_{x}_count',0)) for x in ('schedule','speed','reroute'))))
    lines += [f"| {k} | {old} | {new} |" for k,old,new in comparison]
    lines += ["","## 验证范围","","tests/test_paper_strict_optimization.py覆盖组合、增量ATD/profile/路径、双端no-op、原始species更新、两个fitness尺度、两次FATA、并行复现及历史输出。check_paper_strict.py任一失败返回非零；检查不证明作者未公开实现或实验数值已复原。",""]
    lines += ["", "## Stage1固定关键计划覆盖", "", stage1_coverage_observation(out, s), ""]
    (out/"paper_strict_implementation_report.md").write_text('\n'.join(lines),encoding="utf-8")


def write_paper_html_report(out: Path,s):
    def metrics(keys):
        return '<table>'+''.join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(s.get(k,'')))}</td></tr>" for k in keys)+'</table>'
    def csv_table(name):
        return '<div class="table-scroll">'+pd.read_csv(out/name).to_html(index=False,border=0,escape=True)+'</div>'
    chart=base64.b64encode((out/"paper_convergence.png").read_bytes()).decode("ascii")
    style="body{margin:0;color:#242424;background:white;font:15px/1.6 'Segoe UI','Microsoft YaHei',sans-serif;letter-spacing:0}main{max-width:1060px;margin:auto;padding:24px 16px}h1{font-size:24px}h2{font-size:19px;margin:28px 0 12px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #ddd;text-align:left;padding:8px;overflow-wrap:anywhere}th{color:#385d4a;background:#f5f7f6}.table-scroll{overflow:auto}img{width:100%;height:auto}a{color:#176b50}"
    body=["<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Paper-strict FATA 运行报告</title>",f"<style>{style}</style><main><h1>Paper-strict FATA 运行报告</h1>","<p>普通CI前10%关键计划组合优化 → 重新探测 → 原始ADM采样 → Stage1基础上的FATA增量优化。</p>","<h2>参数与冲突点</h2>",metrics(["paper_objective_scale_mode","seed","NP","Ngen_max_stage1","Ngen_max_stage2","n_jobs","stage1_initial_conflict_points","stage1_final_conflict_points","stage2_final_conflict_points"]),"<h2>最终目标与运行时间</h2>",metrics(["final_Tdelay","final_Tair","final_ORISK","final_n_delay","final_n_battery","final_paper_fitness","changed_flight_count_two_stage","advanced_flight_count","delayed_flight_count","mean_absolute_atd_shift","max_absolute_atd_shift","stage1_fata_time","stage2_fata_time","total_runtime"]),"<h2>两阶段收敛</h2>",f"<img src='data:image/png;base64,{chart}' alt='两阶段 paper fitness 收敛曲线'>"]
    for title,name in (("Stage1策略分配","paper_stage1_strategy_assignment.csv"),("Stage2冲突点策略","paper_stage2_conflict_strategy.csv"),("Flight策略历史","paper_flight_strategy_history.csv")):
        body += [f"<h2>{title}</h2>",csv_table(name)]
    body += ["<p>P为终代概率，selected_strategy为保存最佳原始species标签，并非终代argmax；两端开放，不指定actor。</p><h2>定性观察</h2><ul>"]+['<li>'+html.escape(v)+'</li>' for v in qualitative_observations(s)]+["</ul><h2>实现假设</h2><p>目标论文允许组合，单UAV单策略限制已停用。作者无量纲化未知，raw默认直算式(51)，initial-reference仅实验。via-cell、no-op开关、速度弧长投影、共享窗口、dominant_fraction及ADM/FATA耦合仍含实现假设。</p><p><a href='paper_strict_implementation_report.md'>逐项报告</a> · <a href='paper_convergence.csv'>收敛数据</a> · <a href='adm_probability_history.csv'>ADM历史</a> · <a href='strategy_overview.html'>三维结果</a></p></main></html>"]
    body[-1] = body[-1].replace('</main>', '<h2>Stage1关键计划覆盖</h2><p>'+html.escape(stage1_coverage_observation(out, s))+'</p></main>')
    (out/"fata_run_report.html").write_text('\n'.join(body),encoding="utf-8")
