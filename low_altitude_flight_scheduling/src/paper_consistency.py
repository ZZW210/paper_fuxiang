"""Behavioral checks and run-specific method reports, not claims of author-code parity."""
from __future__ import annotations

import ast
import base64
import copy
import html
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import paper_optimization as model, adm_matching, fata, paper_scheduler
from .conflict_detection import detect_conflicts
from .conflict_network import select_paper_key_flights
from .flight_plan import FlightPlan, compute_eta_times
from .grid import AirspaceGrid


ASSUMPTIONS = [
    "目标论文3.2.1优先：Stage1对每个关键计划分配一个具体策略及对应变量并同步求解；"
    "最终多策略只来自不同Stage2冲突匹配和两阶段累计，不移植参考[6]的全局单UAV单策略限制。",
    "目标论文和参考[6]未完整公开conflict pair actor selection源代码。TRC第9页描述依据剩余航程、"
    "目的地选择改航对象的原则及例子，速度示例还会调整双方，但没有统一可执行的选择算法。"
    "按本轮要求采用二元actor_gene∈[0,2]，<1选plan_a，否则选plan_b，由FATA搜索；这是实现假设而非论文参数。",
    "FATA使用固定维度去重槽：每架UAV一个ATD，每实际航段一个speed，每局部区域一个via块。"
    "每个候选先按原始sampled species和actor构造flight_strategy_requirements，仅对应槽生效；"
    "不因多个冲突复制整套FlightPlan变量，也不重写采样标签。",
    "局部改航编码未公开。本实现用路径索引窗口半径4，reroute_merge_window=5合并相近区域，"
    "重叠窗口也合并以保证拼接；ADM仍逐冲突点。根据区域端点和冲突cell建立局部via包围盒，"
    "默认XY余量5格、Z余量1层；基础A*保持原样，连接路径本身不额外裁剪在该via包围盒内。",
    "采样reroute且选中actor即执行via+A*，没有二次enable开关或特殊原窗口中点旁路。"
    "A*可产生与原路径相同的合法几何结果，不等于禁用策略。改航段速度按归一化弧长继承当前速度，"
    "保留Stage1及当前候选已施加的speed调整；应用顺序是schedule、speed、reroute，再重算ETA、风险和冲突。",
    "dominant_fraction=.20未在可获得文本中明确给出，仍是实现假设。learning_rate=.5来自参考[6]；"
    "P初始化均匀，每次更新用原始优势species，normalize前仅以1e-12稳定epsilon防止数值塌缩，无0.05探索下限。",
    "作者无量纲化细节未知；raw_equation默认直算打印式(49)-(51)，"
    "initial_reference_experimental仍仅作实现假设对照，固定初始参考不变，绝不自动选优或调系数拟合。",
    "初始场景、OD生成、风险地图、基础A*、时间不确定性模型、复杂网络与普通CI不改。"
    "已有严格运行ETD=0提升到1秒的合法初始处理保留，normal保持50/200/200，quick仅20/50/50。",
    "性能计时是包含关系：worker计算时间求和可与父进程等待时间重叠，percentage可超过100%，"
    "不能加总为互斥耗时。multiprocessing_serialization是实际任务pickle探针估计，"
    "不是executor内部序列化或IPC时间的精确拆分。",
]


def run_consistency_checks(cfg):
    checks=[]
    def check(name,value):
        checks.append((name,bool(value)))
    path=[(i,3,1) for i in range(1,7)]
    grid=AirspaceGrid(shape=(8,8,4),obstacles=np.zeros((8,8,4),bool))
    risk=np.ones(grid.shape)
    plans=[FlightPlan(i,path[0],path[-1],list(path),1000,compute_eta_times(path,1000,10,grid.cell_size),[10.0]*5,6.0,50.0) for i in range(2)]
    conflicts=detect_conflicts(plans,cfg)
    layout=model.build_stage1_decision_layout(plans,[0],conflicts,cfg,grid)
    block=layout.blocks[0]
    x=(layout.lower+layout.upper)/2
    x[block.strategy_gene]=1
    x[block.speed_genes]=11.374
    stage1=model.decode_stage1_solution(x,plans,layout,cfg,grid,risk)
    metrics=pd.DataFrame(dict(flight_id=range(100),collective_influence=range(100),weighted_collective_influence=range(100,0,-1)))
    check("ordinary CI fixed top 10",select_paper_key_flights(metrics,100)==list(range(99,89,-1)))
    check("Stage1 one concrete strategy and corresponding variables",model.flight_strategies(stage1[0])==(1,) and stage1[0].atd==1000)
    check("continuous speed and exact segment ETA",stage1[0].speed_profile[0]==11.374 and np.isclose(stage1[0].eta_times[1],1000+100/11.374))
    check("continuous early/late ATD bounds",model.paper_atd_bounds(plans[0],cfg)==(1,2800))
    remaining=detect_conflicts(stage1,cfg)
    l2=model.build_stage2_decision_layout(stage1,remaining,cfg,grid,plans)
    v=(l2.lower+l2.upper)/2
    v[l2.actor_genes]=0
    for b in l2.blocks:
        for genes in b.reroute_genes:
            v[genes]=[4,5,2]
    final=model.decode_stage2_solution(v,stage1,plans,l2,np.full(len(remaining),2),cfg,grid,risk)
    check("Stage2 keeps Stage1 speed and naturally adds reroute",model.flight_strategies(final[0])==(1,2) and final[0].rerouted and all(s==11.374 for s in final[0].speed_profile))
    check("actor selects a single endpoint without parity",final[1] is stage1[1] and l2.actor_genes.stop==len(remaining))
    check("no reroute second switch",not hasattr(l2.blocks[0],"reroute_enable_genes"))
    check("deduplicated ATD/segments",len(l2.blocks)==2 and all(len(b.segment_indices)==len(set(b.segment_indices)) for b in l2.blocks))
    p=adm_matching.initialize_probability_matrix(3)
    check("ADM uniform point matrix",p.shape==(3,3) and np.array_equal(p,np.full((3,3),1/3)))
    check("ADM update learning=.5",cfg["adm"]["learning_rate"]==.5 and np.allclose(adm_matching.update_probability_matrix(p,np.full((1,3),2)),[[1/6,1/6,2/3]]*3))
    for _ in range(250):
        p=adm_matching.update_probability_matrix(p,np.full((1,3),2))
    check("probability numeric stability only",np.all(p>0) and np.allclose(p.sum(axis=1),1) and p[0,0]<1e-10)
    adm_source=inspect.getsource(adm_matching)
    check("probability uses original sampled species","dominant_species = np.asarray(contexts)[ranked]" in adm_source and "return sample_strategy_species(probability, size, rng)" in adm_source)
    check("no global single-flight strategy reconciliation","enforce_single_strategy_per_flight" not in adm_source)
    c=dict(Tdelay=200,Tair=100,ORISK=5,Nc=2,n_delay=2,n_battery=1)
    ref=model.PaperReference(1800,100,5,2)
    raw=copy.deepcopy(cfg)
    raw["optimization"]["paper_objective_scale_mode"]="raw_equation"
    check("raw printed fitness and fixed penalties",np.isclose(model.paper_fitness(c,ref,raw,200,200),.2*(.25*200+.25*100+.5*5)+.8*.9*2*100+1200))
    exp=copy.deepcopy(raw)
    exp["optimization"]["paper_objective_scale_mode"]="initial_reference_experimental"
    check("experimental objective retained",np.isclose(model.paper_fitness(c,ref,exp,200,200),1200+.2*(.25*200/1800+.25+.5)+.8*.9))
    check("dynamic delta and gamma",model.conflict_weight_delta(200,200)==.9 and cfg["fata"]["gamma"]==5)
    check("both stages run improved FATA","fata.fata_optimize_paper(" in adm_source and "adm_fata_optimize(" in inspect.getsource(paper_scheduler))
    source=inspect.getsource(model)+'\n'+adm_source+'\n'+inspect.getsource(paper_scheduler)
    calls={n.func.id if isinstance(n.func,ast.Name) else n.func.attr for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Call) and isinstance(n.func,(ast.Name,ast.Attribute))}
    check("no engineering repair rollback",not calls.intersection({"greedy_time_deconfliction","repair_conflicts","try_apply_action_with_rollback","independent_matching_deconfliction"}))
    seen=[]
    fata.fata_optimize_paper(lambda x,g: seen.append(x.copy()) or float(np.sum(x*x)),[0,0],[1,1],2,population=4,max_iter=1,n_jobs=1)
    check("exact good point initialization",np.array_equal(np.array(seen),fata.good_point_set(4,2,np.zeros(2),np.ones(2))))
    check("no Gaussian additions",".normal(" not in inspect.getsource(fata.fata_optimize_paper))
    check("normal NP and generations unchanged",tuple(cfg["fata"][k] for k in ("NP","Ngen_max_stage1","Ngen_max_stage2"))==(50,200,200))
    return checks


def qualitative_observations(s):
    return [f"冲突点：{s.get('stage1_initial_conflict_points')} -> {s.get('stage1_final_conflict_points')} -> {s.get('stage2_final_conflict_points')}。",
            f"Stage1每架仅一个策略；最终组合UAV={s.get('stage2_strategy_combination_count',0)}（Stage2自身组合，不含仅跨阶段新增组合）。",
            f"实际改航：Stage1={s.get('stage1_actual_rerouted_flight_count',0)}，Stage2相对Stage1={s.get('stage2_actual_rerouted_flight_count',0)}。",
            f"changed={s.get('changed_flight_count_two_stage')}，Stage2增量修改={s.get('stage2_modified_flight_count')}；这些数量不直接入目标，不强制少改动或零冲突。",
            f"提前/延后={s.get('advanced_flight_count')}/{s.get('delayed_flight_count')}；平均/最大绝对ATD偏移={s.get('mean_absolute_atd_shift')}/{s.get('max_absolute_atd_shift')}秒。",
            f"n_delay=0且Tdelay>20000诊断={s.get('large_advance_without_delay_observed')}；提前也累积到绝对Tdelay，不触发限制或repair。"]


def stage1_coverage_observation(out,s):
    conflicts=pd.read_csv(out/"conflicts_uncertain.csv")
    keys=set(pd.read_csv(out/"paper_stage1_strategy_assignment.csv")["flight_id"])
    related=conflicts.plan_a.isin(keys)|conflicts.plan_b.isin(keys)
    return f"固定关键计划涉及初始{int(related.sum())}/{len(conflicts)}点；未涉及的{int((~related).sum())}点是Stage1整体Nc下限，不改CI扩大覆盖。"


def write_implementation_report(out,s):
    cfg=json.loads((out/"paper_run_config.json").read_text(encoding="utf-8"))
    run=cfg.get("run",{})
    lines=["# Paper-strict implementation report","",f"Run ID: {s.get('run_id',run.get('run_id'))}",f"Git commit: {run.get('git_commit','unknown')}","",
           "## 修改文件","","run_main.py、paper_optimization.py、adm_matching.py、fata.py、conflict_detection.py、config.yaml；"
           "paper_scheduler.py仅调度编排与诊断，新增run_archive.py及paper_performance.py；make_fata_report.py改为按run_id读取归档；测试、比较脚本、README与本报告生成器同步更新。"
           "optimization_model.py无需改动，strict不调用legacy repair。","",
           "## Stage1策略逻辑","","对普通CI固定前10%的每个关键计划分配一个具体策略及其对应决策变量，同步求解。"
           "strategy_gene连续[0,3]取floor并限制到0/1/2；每个候选只启用一个主要策略，非三个activation genes。","",
           "## Stage2冲突级ADM","","一个remaining conflict POINT对应P的一行，每代同时独立采样全部冲突的原始标签。"
           "优势种群频率不经过flight投票或标签合并，learning=.5；改进FATA再运行一次。","",
           "## 最终多策略与actor","","策略组合来自多个冲突选择同一actor但匹配不同策略，或来自两阶段累计。"
           "actor_gene<1选plan_a，>=1选plan_b，由FATA搜索，不按奇偶、ID或中心性指定。"
           "对应已选策略变量可以取原值；没有额外reroute_enable。","",
           "## 继承Stage1","","CandidatePlanView由Stage1不可变base引用和选中actor的副本构成；"
           "按schedule、speed、reroute固定顺序施加修改，未涉及航段/计划保留Stage1值。"
           "最终Tdelay和changed对照initial，Stage2增量对照Stage1。","",
           "## 性能与等价性","","| 优化 | 等价性验证 |","|---|---|",
           "| 增量冲突检测，静态pair/occupancy缓存 | 不确定及确定两种检测各50候选，比较所有Conflict字段及顺序 |",
           "| 静态目标贡献缓存 | 50候选按原计划/栅格求和顺序，full/cached fitness差<1e-10 |",
           "| base+actor overlays，无整组deepcopy | 未修改计划保持reference；测试原计划不被修改、跨阶段继承 |",
           "| 每进程持久LRU32768，按地图/风险/权重/端点/via区分key | cached/uncached路径与fitness一致；测试LRU逐项淘汰及环境隔离 |",
           "| 每阶段一个持久process pool，最多8进程，BLAS各1线程 | 同seed串行/并行ADM最佳向量、标签、P、收敛一致 |",
           "| FATA坐标向量化，保留交错随机数和逐行更新 | 3/40/180维与标量更新逐位一致，另检验真实ADM开关一致 |",
           "| 局部via范围、actor与区域合并 | 是本轮方法/编码变更，不宣称与上轮全空域双端编码等价 |","",
           "性能profile采用包含性计时，不是互斥分段；序列化行是pickle探针估计。不得以减NP、Ngen、冲突点或禁用改航提速。","",
           "## 实现假设",""]+[f"{i}. {value}" for i,value in enumerate(ASSUMPTIONS,1)]
    lines += ["","## 本次quick或normal结果","",f"quick={cfg.get('run',{}).get('quick','see run_manifest.json')}；只有实际运行结果，不自动运行normal对照。",""]
    for k in ("run_id","seed","NP","Ngen_max_stage1","Ngen_max_stage2","n_jobs","paper_objective_scale_mode",
              "stage1_dimension","stage2_dimension","stage1_fitness_evaluations","stage2_fitness_evaluations",
              "final_Tdelay","final_Tair","final_ORISK","final_n_delay","final_n_battery","final_paper_fitness",
              "astar_calls","astar_cache_hit_rate","runtime_stage1","runtime_stage2","total_runtime"):
        lines.append(f"- {k}: {s.get(k)}")
    lines += [""]+["- "+v for v in qualitative_observations(s)]+["",stage1_coverage_observation(out,s),"",
              "## 可追溯输出","","每个CSV第一列run_id；PNG底部Run ID并放figures；"
              "run_manifest.json、config_snapshot.yaml和stdout.log记录实际参数与环境；成功后append run_index并更新latest_run。"
              "默认拒绝已存在run目录，只有显式--overwrite-run覆盖。比较必须使用两个child run_id和独立comparison_id。","",
              "## 尚无法确认","","作者的局部变长路径编码、actor通用算法、dominant_fraction、无量纲化、"
              "独立匹配与FATA的完整源代码未公开，不能声称逐行或私有场景数值复原。","",
              "## 验证记录","","本次执行pytest和一次独立quick；详细验证记录见logs/verification.json（由本次开发验证写入）。",""]
    (out/"implementation_assumptions.md").write_text("# Implementation assumptions\n\n"+'\n\n'.join(ASSUMPTIONS)+"\n",encoding="utf-8")
    (out/"paper_strict_implementation_report.md").write_text('\n'.join(lines),encoding="utf-8")


def write_paper_html_report(out,s):
    chart=out/"paper_convergence.png"
    if not chart.exists():
        chart=out/"figures"/chart.name
    data=base64.b64encode(chart.read_bytes()).decode("ascii")
    rows=''.join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(s.get(k,'')))}</td></tr>" for k in
                 ("run_id","paper_objective_scale_mode","NP","Ngen_max_stage1","Ngen_max_stage2","stage1_initial_conflict_points",
                  "stage1_final_conflict_points","stage2_final_conflict_points","changed_flight_count_two_stage","final_Tdelay","final_paper_fitness"))
    style="body{font:15px/1.6 'Segoe UI','Microsoft YaHei',sans-serif;color:#222;background:white;margin:0;letter-spacing:0}main{max-width:1060px;margin:auto;padding:24px 16px}h1{font-size:24px}h2{font-size:19px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;border-bottom:1px solid #ddd;padding:7px;overflow-wrap:anywhere}th{background:#f3f7f4}.scroll{overflow:auto}img{width:100%;height:auto}a{color:#176b50}"
    body=[f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>FATA Run Report</title><style>{style}</style><main><h1>Paper-strict FATA运行报告</h1>",
          f"<table>{rows}</table><h2>两阶段收敛</h2><img src='data:image/png;base64,{data}' alt='FATA convergence'>"]
    for title,name in (("Stage1具体策略","paper_stage1_strategy_assignment.csv"),("最终策略历史","paper_flight_strategy_history.csv"),("Stage2保存最佳species","paper_stage2_conflict_strategy.csv")):
        frame=pd.read_csv(out/name)
        if "record_type" in frame:
            frame=frame[frame.record_type=="selected_best"]
        body.append(f"<h2>{title}</h2><div class='scroll'>{frame.to_html(index=False,border=0)}</div>")
    body += ["<h2>结果诊断</h2><ul>"]+[f"<li>{html.escape(v)}</li>" for v in qualitative_observations(s)]+["</ul>",
              "<p>Stage1每个关键计划分配一个具体策略及对应变量并同步求解；Stage2原始冲突级匹配与FATA actor决策自然形成最终组合。actor和局部via编码仍是实现假设。</p>",
              "<p><a href='paper_strict_implementation_report.md'>方法报告</a> · <a href='run_manifest.json'>Run manifest</a> · <a href='performance_profile.csv'>性能记录</a> · <a href='strategy_overview.html'>三维结果</a></p></main></html>"]
    (out/"fata_run_report.html").write_text('\n'.join(body),encoding="utf-8")
