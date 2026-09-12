"""Controlled upstream experiments only: no scheduler, FATA, ADM or objectives."""
from __future__ import annotations

from collections import Counter
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import resolve_scene_seeds
from .conflict_detection import detect_conflicts, write_conflicts_csv
from .flight_plan import plan_paper_random_traffic, sample_paper_random_traffic
from .grid import AirspaceGrid
from .risk_map import generate_risk_map
from .run_archive import generate_network_run_id, git_metadata
from .scene_diagnostics import analyze_initial_conflict_network, tag_run_csvs
from .utils import path_distance_m

EXPERIMENT_VARIANTS = (
    ("old_gaussian", "meter"),
    ("reference28_gravity", "meter"),
    ("reference28_gravity", "grid"),
    ("reference28_gravity", "kilometer"),
)
PAPER_REFERENCES = dict(det_Nc=53, unc_Nc=97, ci_top10_coverage=78/97)


def array_fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def traffic_records(tasks):
    return [dict(flight_id=t.id, start_x=t.start[0], start_y=t.start[1], start_z=t.start[2],
                 goal_x=t.goal[0], goal_y=t.goal[1], goal_z=t.goal[2], etd=t.etd, speed=t.speed)
            for t in tasks]


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_route_outputs(plans, grid, run_id, directory):
    xy = np.zeros(grid.shape[:2], dtype=int)
    reuse = Counter(cell for plan in plans for cell in set(plan.path))
    reuse_map = np.zeros(grid.shape, dtype=int)
    for cell, count in reuse.items():
        reuse_map[cell] = count
    for plan in plans:
        for x, y in {(p[0], p[1]) for p in plan.path}:
            xy[x, y] += 1
    np.save(directory / "route_cell_reuse.npy", reuse_map)
    np.save(directory / "route_density_xy.npy", xy)
    pd.DataFrame([dict(run_id=run_id, grid_x=c[0], grid_y=c[1], grid_z=c[2],
                       distinct_flight_count=count) for c, count in sorted(reuse.items())]).to_csv(
        directory / "route_cell_reuse.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(xy.T, origin="lower", cmap="magma",
                   extent=(0, grid.shape[0]*grid.cell_size[0], 0, grid.shape[1]*grid.cell_size[1]))
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Distinct flight footprints per horizontal cell")
    fig.colorbar(im, ax=ax, label="flight count")
    fig.text(0.01, 0.005, f"Run ID: {run_id}", fontsize=6)
    fig.tight_layout()
    fig.savefig(directory / "route_density_heatmap.png", dpi=140)
    plt.close(fig)
    with (directory / "initial_plans.pkl").open("wb") as fh:
        pickle.dump(plans, fh)


def write_comparison_report(rows, directory, comparison_id):
    a, b, c, d = rows
    table_fields = ["group", "det_Nc", "unc_Nc", "network_edges", "max_degree", "degree_gini",
                    "ci_top10_coverage", "continuous_conflict_ratio", "route_density_gini",
                    "route_cell_reuse_p90", "route_cell_reuse_max", "routes_different_from_A"]
    table = "| " + " | ".join(table_fields) + " |\n| " + " | ".join("---" for _ in table_fields) + " |\n"
    for row in rows:
        table += "| " + " | ".join(f"{row[k]:.6f}" if isinstance(row[k], float) else str(row[k])
                                    for k in table_fields) + " |\n"
    metrics = ["route_density_gini", "route_cell_reuse_p90", "route_cell_reuse_max",
               "continuous_conflict_ratio", "max_degree", "degree_gini", "ci_top10_coverage"]
    deltas = "\n".join(f"- {key}: A={a[key]:.6f}, B={b[key]:.6f}, B−A={b[key]-a[key]:+.6f}"
                       for key in metrics)
    pop_effect = abs(b["ci_top10_coverage"]-a["ci_top10_coverage"])
    scale_effect = max(r["ci_top10_coverage"] for r in rows[1:])-min(r["ci_top10_coverage"] for r in rows[1:])
    if pop_effect == scale_effect == 0:
        attribution = "两种因素在本样本中均未改变 CI Top10 coverage，不能据此归因其与论文的差距。"
    else:
        larger = "A* 距离尺度" if scale_effect > pop_effect else "人口模型的 meter 对照"
        attribution = (f"本样本中，{larger} 的 coverage 变化较大。人口模型 |B−A|={pop_effect:.6f}，"
                       f"gravity 三种尺度的 coverage 范围={scale_effect:.6f}。这是单座合成城市、单个交通样本的"
                       "条件比较，不是总体因果结论，也不排除未公开城市布局、计数约定和风险近似的影响。")
    concentrated = "增加" if b["route_density_gini"] > a["route_density_gini"] else "减少" if b["route_density_gini"] < a["route_density_gini"] else "未变化"
    report = f"""# 人口风险—A*—冲突网络结构比较

Comparison ID: `{comparison_id}`

环境种子={a['environment_seed']}；交通种子={a['traffic_seed']}；飞行计划数={a['n_flights']}。
四组共享完全相同的建筑数组、OD、ETD 和初始速度；各组重新运行 A*，不重抽交通。
参数 beta={b['beta']}，中心数设定={b['requested_population_centers']}，实际={b['actual_population_centers']}。
没有执行 Stage1、Stage2、FATA、ADM 或调度目标函数，没有按公开结果选择参数。

A: old_gaussian + meter（仅旧人口生成器对照；同一风险计算与归一化）。
B/C/D: reference28_gravity + meter/grid/kilometer。
默认尺度仍是 meter；全部使用 alpha_r=0.8、alpha_L=0.2。

## 论文公开值（仅 comparison）

deterministic conflicts=53；uncertain conflicts=97；CI Top10 point coverage=78/97≈0.804124。
来源：[目标论文](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2025.31479)。
这些数值不参与中心检测、beta 选择、路径规划、CI 排序或种子选择。

{table}

## 1. Gravity 是否使航迹更集中？

固定 meter 的 A→B，水平 route density Gini {concentrated}；同时有
{b['routes_different_from_A']}/{b['n_flights']} 条路径与 A 不同。
该指标包括整张水平图的零使用网格。需结合下面的 3D 复用、连续冲突和节点集中度，
不能仅凭 Gini 或视觉热图宣称论文网络已复现。

## 2. 复用、连续冲突、度集中与 Top10 覆盖变化

{deltas}

3D reuse 是每个访问过的 3D 网格的不同飞行计划数（同一航迹多次访问只计一次）；
p90 只在已访问 3D 网格上计算。Gini 则是整张水平图上的不同航迹 footprint 计数，
两者不是同一个统计量。连续冲突沿用原有邻接定义，比例为多事件连续段内事件数/全部不确定冲突事件数。
CI 沿用原公式、l=2、固定 Top10 和 flight-id tie break；点覆盖统计 incident event union。

## 3. A* 距离尺度的影响

在同一 gravity 人口层中，B→C→D 的水平步长分别是 100m、1 grid unit、0.1km；
垂直和斜向移动按原有物理几何同比缩放，g 的长度项与 h 同时换单位，风险不换尺度。
原有 w(n)、26 邻域、关闭节点及终止条件不变。各模式与 A 的路径差异数见表。
coverage 范围={scale_effect:.6f}；uncertain conflicts 范围=
{min(r['unc_Nc'] for r in rows[1:])}..{max(r['unc_Nc'] for r in rows[1:])}。
runtime 包含各组诊断图/文件输出，不含共享城市和交通采样；不能作为纯 A* 性能计时。

## 4. Top10 coverage 差距主要来自哪里？

{attribution}
四组 coverage 范围仅 {min(r['ci_top10_coverage'] for r in rows):.6f}..
{max(r['ci_top10_coverage'] for r in rows):.6f}，全部远低于论文 0.804124。
人口模型对照和距离尺度的 coverage 变化幅度，都远小于这一缺口；
因此不能确认低 Top10 coverage 的主要成因已定位，也不能把更多冲突/更高航迹复用误判为更接近论文结构。
风险数值仍使用项目既有下坠/漂移近似与全图 min-max 归一化；归一化方法也是未公开实现假设，
本轮没有同时扫描它。当前实验不能单独排除这一上游因素。

## 5. 公开参数与实现假设

目标论文明确：3500 人/km²、仿真建筑地图、综合对地风险结构、100 条约 6km 初始航迹、
10m/s、风险/距离权重0.8/0.2；公开网络值仅用于以上对照。
实现假设：4 个中心、10×10 窗口、1km NMS 间距、beta=2、建筑密度归一化与平局规则、
静态 snapshot、既有风险近似与 min-max、A* 的距离单位、CI l=2 和连续段诊断口径。
参考 [28]：[作者仓库](https://hdl.handle.net/10356/170158)，
[DOI](https://doi.org/10.1109/DASC58513.2023.10311224)。已核验本地
`Manuscript_DASC_2023.pdf` 第 2 页 III.B 节公式 (1)–(2)：最近中心、1km 内指数扩散、
1km 外静态人口。原文未规定恰好 1km 的分支，此处按用户规格取静态基准。
不复制新加坡人口密度，不使用 MRT 数据或随机森林，也不替换目标论文的综合风险公式。
完整假设见每个 run 的 `implementation_assumptions.md`。

## 独立运行目录

"""
    report += "\n".join(f"- {r['group']}: [完整输出](../../runs/{r['run_id']}/)，run_id=`{r['run_id']}`"
                        for r in rows)
    (directory / "risk_route_network_report.md").write_text(report + "\n", encoding="utf-8")


def run_structure_experiments(cfg, output_root, environment_seed=None, traffic_seed=None,
                              n_flights=100, beta=None, project_root=None):
    """Create four never-overwritten runs plus an isolated comparison directory."""
    cfg = copy.deepcopy(cfg)
    if cfg["optimization"]["scheduler_mode"] != "paper_strict":
        raise ValueError("This controlled experiment requires paper_strict config")
    env, traffic_seed = resolve_scene_seeds(cfg, environment_seed=environment_seed, traffic_seed=traffic_seed)
    if n_flights < 1:
        raise ValueError("n_flights must be positive")
    cfg["paper_scene"]["astar"].update(risk_weight=0.8, distance_weight=0.2)
    if beta is not None:
        cfg["population"]["beta"] = beta
    project_root = Path(project_root or Path(__file__).resolve().parents[1])
    output_root = Path(output_root)
    commit, dirty = git_metadata(project_root.parent)
    comparison_id = generate_network_run_id("comparison", "fixed_traffic", env, traffic_seed, commit)
    comparison_dir = output_root / "comparisons" / comparison_id
    comparison_dir.mkdir(parents=True, exist_ok=False)
    comparison_manifest = dict(comparison_id=comparison_id, environment_seed=env, traffic_seed=traffic_seed,
                               git_commit=commit, git_dirty=dirty, n_flights=n_flights, status="running",
                               variants=EXPERIMENT_VARIANTS, completed_runs=[], optimizers_executed=False,
                               paper_references=PAPER_REFERENCES)
    _write_json(comparison_dir / "comparison_manifest.json", comparison_manifest)
    rows, reference_paths = [], None
    try:
        grid = AirspaceGrid.from_config(cfg, seed=env)
        tasks = sample_paper_random_traffic(grid, cfg, n_flights, traffic_seed)
        records = traffic_records(tasks)
        traffic_hash = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
        obstacle_hash = array_fingerprint(grid.obstacles)
        pd.DataFrame([dict(run_id=comparison_id, **r) for r in records]).to_csv(
            comparison_dir / "shared_traffic_tasks.csv", index=False)
        for group, (population_mode, distance_scale) in zip("ABCD", EXPERIMENT_VARIANTS):
            run_cfg = copy.deepcopy(cfg)
            run_id = generate_network_run_id(population_mode, distance_scale, env, traffic_seed, commit)
            directory = output_root / "runs" / run_id
            directory.mkdir(parents=True, exist_ok=False)
            run_cfg.update(population_model=population_mode, astar_distance_scale_mode=distance_scale)
            run_cfg["experiment"] = dict(network_only=True, allow_old_gaussian_baseline=group == "A")
            run_cfg["run"] = dict(run_id=run_id, git_commit=commit, git_dirty=dirty)
            run_cfg["flight_generation"]["n_flights"] = n_flights
            run_cfg["flight"]["n_flights"] = n_flights
            run_cfg["paper_scene"]["n_flights"] = n_flights
            manifest = dict(run_id=run_id, comparison_id=comparison_id, group=group,
                            population_mode=population_mode, distance_scale=distance_scale,
                            environment_seed=env, traffic_seed=traffic_seed, n_flights=n_flights,
                            git_commit=commit, git_dirty=dirty, obstacle_sha256=obstacle_hash,
                            traffic_sha256=traffic_hash, status="running", optimizers_executed=False,
                            stages_executed=[], start_time=datetime.now().astimezone().isoformat(),
                            source_sha256={name: hashlib.sha256((project_root / "src" / name).read_bytes()).hexdigest()
                                           for name in ("risk_map.py", "astar_3d.py", "flight_plan.py", "risk_route_network.py",
                                                        "scene_diagnostics.py", "conflict_detection.py", "conflict_network.py")})
            _write_json(directory / "run_manifest.json", manifest)
            _write_json(directory / "config_snapshot.json", run_cfg)
            _write_json(directory / "scene_config.json", run_cfg)
            started = time.perf_counter()
            try:
                print(f"[{group}] run_id={run_id}", flush=True)
                risk = generate_risk_map(grid, run_cfg, directory)
                plans = plan_paper_random_traffic(grid, risk, run_cfg, tasks)
                if [(p.id, p.start, p.goal, p.etd) for p in plans] != [(t.id, t.start, t.goal, t.etd) for t in tasks]:
                    raise AssertionError("Experiment changed the frozen traffic table")
                if array_fingerprint(grid.obstacles) != obstacle_hash:
                    raise AssertionError("Experiment changed the shared building map")
                uncertain = detect_conflicts(plans, run_cfg, uncertain=True)
                deterministic = detect_conflicts(plans, run_cfg, uncertain=False)
                diag, _ = analyze_initial_conflict_network(plans, uncertain, deterministic, traffic_seed,
                                                           run_id, directory, grid)
                write_conflicts_csv(uncertain, directory / "initial_conflicts_uncertain.csv")
                write_conflicts_csv(deterministic, directory / "initial_conflicts_deterministic.csv")
                pd.DataFrame([dict(run_id=run_id, **r) for r in records]).to_csv(
                    directory / "traffic_tasks.csv", index=False)
                np.save(directory / "obstacles.npy", grid.obstacles)
                _write_route_outputs(plans, grid, run_id, directory)
                paths = [p.path for p in plans]
                if reference_paths is None:
                    reference_paths = paths
                changed = sum(path != ref for path, ref in zip(paths, reference_paths))
                row = dict(run_id=run_id, group=group, population_mode=population_mode,
                           distance_scale=distance_scale, det_Nc=len(deterministic), unc_Nc=len(uncertain),
                           network_edges=diag["network_edges"], max_degree=diag["max_degree"], degree_gini=diag["degree_gini"],
                           ci_top10_coverage=diag["top10_ci_point_coverage"],
                           continuous_conflict_ratio=diag["continuous_conflict_point_ratio"],
                           route_density_gini=diag["route_density_gini"], route_cell_reuse_p90=diag["route_cell_reuse_p90"],
                           route_cell_reuse_max=diag["route_cell_reuse_max"],
                           mean_conflict_points_per_edge=diag["mean_conflict_points_per_edge"],
                           max_conflict_points_per_edge=diag["max_conflict_points_per_edge"],
                           runtime=time.perf_counter()-started, environment_seed=env, traffic_seed=traffic_seed,
                           n_flights=n_flights, beta=float(run_cfg["population"]["beta"]),
                           requested_population_centers=run_cfg["population"]["n_population_centers"],
                           actual_population_centers=len(pd.read_csv(directory / "population_centers.csv")),
                           routes_different_from_A=changed,
                           mean_route_length_m=float(np.mean([path_distance_m(p.path, grid.cell_size) for p in plans])))
                pd.DataFrame([row]).to_csv(directory / "risk_route_network_metrics.csv", index=False)
                tag_run_csvs(directory, run_id)
                manifest.update(status="completed", stages_executed=["grid", "population", "risk", "astar", "flight_plans",
                                                                    "conflict_detection", "network", "CI"],
                                population_map_sha256=array_fingerprint(np.load(directory / "population_map.npy")),
                                risk_map_sha256=array_fingerprint(risk), metrics=row)
                rows.append(row)
                comparison_manifest["completed_runs"].append(run_id)
                pd.DataFrame(rows).to_csv(comparison_dir / "risk_route_network_comparison.csv", index=False)
                print(f"[{group}] det={len(deterministic)} unc={len(uncertain)} "
                      f"CI Top10={row['ci_top10_coverage']:.6f} runtime={row['runtime']:.2f}s", flush=True)
            except Exception as error:
                manifest.update(status="failed", error=str(error))
                raise
            finally:
                manifest.update(end_time=datetime.now().astimezone().isoformat(), elapsed_seconds=time.perf_counter()-started)
                _write_json(directory / "run_manifest.json", manifest)
                _write_json(comparison_dir / "comparison_manifest.json", comparison_manifest)
        write_comparison_report(rows, comparison_dir, comparison_id)
        comparison_manifest["status"] = "completed"
    except Exception as error:
        comparison_manifest.update(status="failed", error=str(error))
        raise
    finally:
        comparison_manifest["end_time"] = datetime.now().astimezone().isoformat()
        _write_json(comparison_dir / "comparison_manifest.json", comparison_manifest)
    return rows, comparison_dir
