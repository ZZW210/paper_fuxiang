"""Reproduce the paper's key-flight-ratio experiment on one fixed scene."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime
import json
from pathlib import Path
import platform
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import load_config, resolve_scene_seeds
from src.conflict_detection import detect_conflicts
from src.conflict_network import build_conflict_network, collective_influence, select_paper_key_flights
from src.flight_plan import plan_paper_random_traffic, sample_paper_random_traffic
from src.grid import AirspaceGrid
from src.paper_scheduler import optimize_paper_schedule
from src.risk_map import generate_risk_map
from src.risk_route_network import array_fingerprint
from src.scene_diagnostics import coverage
from src.utils import set_random_seed

PAPER_TABLE9 = [
    (0.03, 1.2, 1.2, 114.7, 17925.4, 0.60), (0.04, 1.0, 2.0, 113.5, 17999.7, 0.62),
    (0.05, 1.2, 1.6, 119.1, 17887.2, 0.58), (0.06, 0.6, 1.6, 101.9, 15615.6, 0.48),
    (0.07, 0.2, 1.4, 103.1, 14697.2, 0.51), (0.08, 0.0, 0.2, 106.9, 13302.0, 0.44),
    (0.09, 0.0, 0.4, 95.2, 13077.1, 0.39), (0.10, 0.0, 0.5, 97.9, 12322.5, 0.31),
    (0.11, 0.0, 0.4, 96.0, 12418.7, 0.32), (0.12, 0.0, 0.4, 96.4, 12353.1, 0.32),
]


def _analysis_id(environment_seed: int, traffic_seed: int) -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{stamp}_key_ratio_env{environment_seed}_traffic{traffic_seed}"


def _validate_fixed_scene(cfg: dict) -> None:
    if cfg["optimization"].get("scheduler_mode") != "paper_strict":
        raise ValueError("key-ratio analysis requires paper_strict")
    if (cfg.get("population_model"), cfg.get("population_map_mode"), cfg.get("astar_distance_scale_mode")) != ("reference28_gravity", "static_snapshot", "meter"):
        raise ValueError("fixed scene baseline differs")
    if (cfg["population"].get("beta"), cfg["population"].get("n_population_centers")) != (2.0, 4):
        raise ValueError("fixed population baseline differs")
    scene = cfg["paper_scene"]
    if (scene["n_flights"], scene["initial_speed"]["value"], scene["takeoff"]["min"], scene["takeoff"]["max"]) != (100, 10.0, 0, 1800):
        raise ValueError("fixed traffic baseline differs")
    if (scene["astar"]["risk_weight"], scene["astar"]["distance_weight"], cfg["conflict"]["t_conflict"], cfg["conflict"]["alpha"]) != (0.8, 0.2, 30, 0.05):
        raise ValueError("fixed A*/conflict baseline differs")
    if (cfg["fata"]["NP"], cfg["fata"]["Ngen_max_stage1"], cfg["fata"]["Ngen_max_stage2"], cfg["fata"]["Parf"]) != (50, 200, 200, 0.2):
        raise ValueError("Table 9 experiment requires NP=50, Ngen=200 per stage, Parf=0.2")


def initial_ci_ranking(plans, conflicts) -> pd.DataFrame:
    graph = build_conflict_network(plans, conflicts)
    ci = collective_influence(graph, 2)
    degree = dict(graph.degree())
    involvement = Counter()
    for conflict in conflicts:
        involvement.update((conflict.plan_a, conflict.plan_b))
    ordered = sorted(graph.nodes, key=lambda fid: (-ci[fid], fid))
    return pd.DataFrame([dict(rank=index, flight_id=int(fid), degree=int(degree[fid]), CI=float(ci[fid]),
                              conflict_point_involvement=int(involvement[fid]))
                         for index, fid in enumerate(ordered, 1)])


def key_ids_from_ranking(ranking: pd.DataFrame, important_ratio: float, n_flights: int) -> list[int]:
    k = max(1, round(float(important_ratio) * n_flights))
    return ranking.head(k)["flight_id"].astype(int).tolist()


def _changed(before, after) -> bool:
    return (abs(before.atd - after.atd) > 1e-6 or before.path != after.path
            or len(before.speed_profile) != len(after.speed_profile)
            or not np.allclose(before.speed_profile, after.speed_profile, atol=1e-6, rtol=0.0))


def run_one(initial_plans, initial_conflicts, key_ids, cfg, grid, risk_map, ratio: float, optimizer_seed: int) -> tuple[dict, list[dict]]:
    run_cfg = copy.deepcopy(cfg)
    run_cfg["optimization"]["important_ratio"] = float(ratio)
    run_cfg["flight"]["random_seed"] = int(optimizer_seed)
    set_random_seed(optimizer_seed)
    started = time.perf_counter()
    result = optimize_paper_schedule(copy.deepcopy(initial_plans), initial_conflicts, key_ids, run_cfg, grid, risk_map, progress=False)
    elapsed = time.perf_counter() - started
    final_conflicts = detect_conflicts(result.final.plans, run_cfg, uncertain=True)
    initial_risk = sum(float(risk_map[cell]) for plan in initial_plans for cell in plan.path)
    final_risk = sum(float(risk_map[cell]) for plan in result.final.plans for cell in plan.path)
    stage1_changed = sum(_changed(before, after) for before, after in zip(initial_plans, result.stage1.plans))
    stage2_changed = sum(_changed(before, after) for before, after in zip(result.stage1.plans, result.final.plans))
    row = dict(key_ratio=float(ratio), K=len(key_ids), optimizer_seed=int(optimizer_seed),
               remaining_conflicts=len(final_conflicts), delayed_flight_count=int(result.final.components["n_delay"]),
               runtime_sec=elapsed, final_fitness=float(result.final.fitness),
               risk_increase_percent=(final_risk - initial_risk) / initial_risk * 100 if initial_risk else 0.0,
               key_conflict_point_coverage=coverage(initial_conflicts, key_ids)[0],
               key_conflict_pair_coverage=coverage(initial_conflicts, key_ids)[1],
               stage1_remaining_conflicts=len(result.stage1.conflicts), stage2_remaining_conflicts=len(final_conflicts),
               stage1_changed_flights=stage1_changed, stage2_changed_flights=stage2_changed,
               total_changed_flights=sum(_changed(before, after) for before, after in zip(initial_plans, result.final.plans)),
               stage1_runtime=result.stage1_seconds, stage2_runtime=result.stage2_seconds,
               stage1_best_fitness=float(result.stage1.fitness), stage2_executed=bool(result.adm is not None),
               stage1_dimension=result.dimensions[0], stage2_dimension=result.dimensions[1])
    convergence = [dict(key_ratio=float(ratio), optimizer_seed=int(optimizer_seed),
                        iteration=int(item["stage_generation"]), stage=item["stage"], fitness=float(item["fitness"]))
                   for item in result.convergence]
    return row, convergence


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ratio, group in raw.groupby("key_ratio", sort=True):
        row = dict(key_ratio=float(ratio), K=int(group["K"].iloc[0]),
                   key_conflict_coverage=float(group["key_conflict_point_coverage"].iloc[0]),
                   success_rate_zero_conflict=float((group["remaining_conflicts"] == 0).mean()))
        for source, target in (("remaining_conflicts", "remaining_conflicts"), ("delayed_flight_count", "delayed_flights"),
                               ("runtime_sec", "runtime"), ("final_fitness", "final_fitness"),
                               ("risk_increase_percent", "risk_increase")):
            row[f"{target}_mean"] = float(group[source].mean())
            row[f"{target}_std"] = float(group[source].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("key_ratio")


def _plot_convergence(convergence: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    stage2 = convergence[convergence["stage"].eq("stage2")]
    for ratio, group in stage2.groupby("key_ratio", sort=True):
        stats = group.groupby("iteration")["fitness"].agg(["mean", "std"])
        label = f"p={ratio:.2f}"
        ax.plot(stats.index, stats["mean"], label=label, linewidth=1.4)
        ax.fill_between(stats.index, stats["mean"] - stats["std"], stats["mean"] + stats["std"], alpha=.12)
    ax.set(xlabel="Iteration", ylabel="Stage2 best fitness", title="Key-ratio Stage2 convergence")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_comparison(table: pd.DataFrame, reference: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(table["key_ratio"], table["final_fitness_mean"], marker="o", label="reproduction")
    ax.plot(reference["key_ratio"], reference["final_fitness_mean"], marker="s", label="paper Table 9")
    ax.set(xlabel="Key-flight ratio", ylabel="Final fitness", title="Table 9 final-fitness comparison")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_report(path: Path, table: pd.DataFrame, reference: pd.DataFrame, repeats: int) -> None:
    stable = table[table["success_rate_zero_conflict"].eq(1.0)]
    pool = stable if len(stable) else table
    best = pool.sort_values(["final_fitness_mean", "remaining_conflicts_mean", "key_ratio"]).iloc[0]
    monotonic = bool(np.all(np.diff(table["key_conflict_coverage"].to_numpy()) >= -1e-12))
    difference = float(best["key_ratio"] - .10)
    text = f"""# Key Flight Ratio Analysis

Each ratio was independently optimized {repeats} times to reduce the influence of randomized optimization. This does not assert that the paper's Table 9 used 30 repetitions.

## Answers

A. The best key-flight ratio in this fixed scene is {best['key_ratio']:.2f} (K={int(best['K'])}), selected by {'stable zero-conflict success followed by ' if len(stable) else 'lowest mean remaining conflicts followed by '}lowest mean final fitness.

B. It is {'consistent' if abs(difference) < 1e-12 else 'not consistent'} with the paper's 0.10.

C. Difference from 0.10: {difference:+.2f}.

D. Initial CI conflict point coverage is {'monotonically non-decreasing' if monotonic else 'not monotonic'} because each K is a longer prefix of one fixed CI ranking.

E. Higher coverage need not keep improving final optimization: it also enlarges the Stage1 joint decision space and can increase FATA search difficulty, while Stage2 independently assigns strategies to residual conflict points.

F. The result should be interpreted by the small-ratio capacity versus larger-ratio decision-dimension trade-off shown in the measured table, rather than forcing 0.10 to win.

G. The comparison figure preserves raw values. It can compare trends with Table 9, but numerical agreement is not expected because the paper's original random scene and seeds are unavailable.

The paper's 0.10 was obtained in its specific simulation scenario. This reproduction uses a fixed synthetic environment and traffic seed, so a different optimum is allowed.
"""
    path.write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-seed", type=int, default=2025)
    parser.add_argument("--traffic-seed", type=int, default=316)
    parser.add_argument("--ratios", nargs="+", type=float, default=[.03, .04, .05, .06, .07, .08, .09, .10, .11, .12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output-root", default="outputs/key_ratio_analysis")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats < 1 or any(not 0 < ratio <= 1 for ratio in args.ratios):
        raise ValueError("repeats must be positive and ratios must lie in (0,1]")
    root = Path(__file__).resolve().parent
    cfg = load_config(root / "config.yaml", {"optimization": {"scheduler_mode": "paper_strict", "n_jobs": args.n_jobs}})
    resolve_scene_seeds(cfg, environment_seed=args.environment_seed, traffic_seed=args.traffic_seed)
    _validate_fixed_scene(cfg)
    analysis_id = _analysis_id(args.environment_seed, args.traffic_seed)
    out = root / args.output_root / analysis_id
    out.mkdir(parents=True, exist_ok=False)
    grid = AirspaceGrid.from_config(cfg, seed=args.environment_seed)
    risk_map = generate_risk_map(grid, cfg)
    tasks = sample_paper_random_traffic(grid, cfg, 100, args.traffic_seed)
    initial_plans = plan_paper_random_traffic(grid, risk_map, cfg, tasks)
    initial_det = detect_conflicts(initial_plans, cfg, uncertain=False)
    initial_conflicts = detect_conflicts(initial_plans, cfg, uncertain=True)
    if (len(initial_det), len(initial_conflicts)) != (53, 107):
        raise RuntimeError(f"fixed scene is not reproducible: deterministic={len(initial_det)}, uncertain={len(initial_conflicts)}")
    ranking = initial_ci_ranking(initial_plans, initial_conflicts)
    ranking.to_csv(out / "initial_ci_ranking.csv", index=False)
    reference = pd.DataFrame(PAPER_TABLE9, columns=["key_ratio", "remaining_conflicts_mean", "delayed_flights_mean", "runtime_mean", "final_fitness_mean", "risk_increase_mean_percent"])
    reference.to_csv(out / "paper_table9_reference.csv", index=False)
    manifest = dict(analysis_id=analysis_id, environment_seed=args.environment_seed, traffic_seed=args.traffic_seed,
                    ratios=args.ratios, repeats=args.repeats, optimizer_seeds=list(range(args.repeats)), n_jobs=args.n_jobs,
                    initial_deterministic_conflicts=len(initial_det), initial_uncertain_conflicts=len(initial_conflicts),
                    obstacle_hash=array_fingerprint(grid.obstacles), risk_map_hash=array_fingerprint(risk_map),
                    python=sys.version, platform=platform.platform(), parameters=dict(NP=50, Ngen=200, Parf=.2), status="running")
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows, convergence = [], []
    started = time.perf_counter()
    total = len(args.ratios) * args.repeats
    for ratio_index, ratio in enumerate(args.ratios, 1):
        ratio_out = out / f"ratio_{round(ratio * 100):03d}"
        ratio_out.mkdir()
        key_ids = key_ids_from_ranking(ranking, ratio, len(initial_plans))
        if key_ids != select_paper_key_flights(ranking.rename(columns={"CI": "collective_influence"}), len(initial_plans), ratio):
            raise AssertionError("ratio selection differs from fixed initial CI prefix")
        pd.DataFrame({"rank": range(1, len(key_ids) + 1), "flight_id": key_ids}).to_csv(ratio_out / "key_flights.csv", index=False)
        ratio_rows = []
        for optimizer_seed in range(args.repeats):
            row, trace = run_one(initial_plans, initial_conflicts, key_ids, cfg, grid, risk_map, ratio, optimizer_seed)
            rows.append(row); ratio_rows.append(row); convergence.extend(trace)
            print(f"[{(ratio_index-1)*args.repeats + optimizer_seed + 1}/{total}] p={ratio:.2f} seed={optimizer_seed} final={row['remaining_conflicts']}", flush=True)
            pd.DataFrame(rows).to_csv(out / "raw_runs.csv", index=False)
        pd.DataFrame(ratio_rows).to_csv(ratio_out / "raw_runs.csv", index=False)
    raw = pd.DataFrame(rows)
    trace = pd.DataFrame(convergence)
    table = aggregate(raw)
    raw.to_csv(out / "raw_runs.csv", index=False)
    trace.to_csv(out / "convergence_raw.csv", index=False)
    table.to_csv(out / "key_ratio_table9_reproduction.csv", index=False)
    _plot_convergence(trace, out / "key_ratio_convergence.png")
    _plot_comparison(table, reference, out / "key_ratio_table9_comparison.png")
    _write_report(out / "key_ratio_analysis_report.md", table, reference, args.repeats)
    manifest.update(status="completed", total_runtime_sec=time.perf_counter() - started)
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"output: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
