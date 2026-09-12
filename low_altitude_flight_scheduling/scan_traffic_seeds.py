"""Scan paper-strict traffic seeds against one immutable city environment.

This tool deliberately stops after initial A*, conflict detection, and CI analysis.
It does not import or invoke any scheduling/optimization entry point.
"""
from __future__ import annotations

# Set before NumPy is imported (and inherited by spawned Windows workers).
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import networkx as nx
import numpy as np
import pandas as pd

from src.config import load_config, resolve_scene_seeds
from src.conflict_detection import detect_conflicts, write_conflicts_csv
from src.conflict_network import build_conflict_network, collective_influence, plot_network
from src.flight_plan import plan_paper_random_traffic, sample_paper_random_traffic
from src.grid import AirspaceGrid
from src.risk_map import generate_population_density, generate_risk_map
from src.risk_route_network import array_fingerprint, traffic_records
from src.run_archive import git_metadata, validate_run_id
from src.scene_diagnostics import analyze_initial_conflict_network, coverage, route_concentration_metrics, tag_run_csvs

TARGET_COVERAGE = 78 / 97
_WORKER: dict[str, object] = {}


def scan_id(environment_seed: int, seed_start: int, seed_end: int, commit: str) -> str:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    value = f"{timestamp}_traffic_scan_env{environment_seed}_seed{seed_start}-{seed_end}_{commit[:8]}"
    validate_run_id(value)
    return value


def validate_baseline(cfg: dict) -> None:
    expected = {
        "scheduler_mode": "paper_strict", "population_model": "reference28_gravity",
        "population_map_mode": "static_snapshot", "astar_distance_scale_mode": "meter",
    }
    actual = {
        "scheduler_mode": cfg["optimization"].get("scheduler_mode"),
        "population_model": cfg.get("population_model"),
        "population_map_mode": cfg.get("population_map_mode"),
        "astar_distance_scale_mode": cfg.get("astar_distance_scale_mode"),
    }
    if actual != expected:
        raise ValueError(f"traffic scan requires the fixed paper baseline: {actual}")
    population = cfg["population"]
    scene = cfg["paper_scene"]
    if (population.get("window_size"), population.get("min_center_spacing_m"), population.get("n_population_centers"), population.get("beta")) != (10, 1000, 4, 2.0):
        raise ValueError("population baseline is not window=10, spacing=1000m, centers=4, beta=2")
    if (scene.get("n_flights"), scene["initial_speed"].get("value"), scene["takeoff"].get("min"), scene["takeoff"].get("max")) != (100, 10.0, 0, 1800):
        raise ValueError("paper traffic baseline is not 100 flights, speed=10, uniform ETD=[0,1800]")
    if (scene["astar"].get("risk_weight"), scene["astar"].get("distance_weight"), cfg["conflict"].get("t_conflict"), cfg["conflict"].get("alpha")) != (0.8, 0.2, 30, 0.05):
        raise ValueError("A*/conflict baseline differs from the fixed paper settings")
    if (cfg["network"].get("ci_l"), cfg["network"].get("top_k"), cfg["optimization"].get("stage1_key_ratio")) != (2, 10, 0.10):
        raise ValueError("CI baseline must be l=2, Top10, fixed 10%")


def environment_hashes(grid, population: np.ndarray, risk: np.ndarray) -> dict[str, str]:
    buildings = array_fingerprint(grid.obstacles)
    population_hash = array_fingerprint(population)
    risk_hash = array_fingerprint(risk)
    digest = hashlib.sha256(f"{buildings}:{population_hash}:{risk_hash}".encode("ascii")).hexdigest()
    return dict(building_map_hash=buildings, population_map_hash=population_hash, risk_map_hash=risk_hash, environment_hash=digest)


def prepare_environment(cfg: dict, environment_seed: int):
    grid = AirspaceGrid.from_config(cfg, seed=environment_seed)
    population = generate_population_density(grid, cfg)
    risk = generate_risk_map(grid, cfg, output_dir=None)
    return grid, risk, environment_hashes(grid, population, risk)


def _init_worker(cfg: dict, environment_seed: int, expected_hashes: dict[str, str]) -> None:
    grid, risk, hashes = prepare_environment(cfg, environment_seed)
    if hashes != expected_hashes:
        raise AssertionError("worker environment hash differs from the fixed scan environment")
    _WORKER.update(cfg=cfg, grid=grid, risk=risk, hashes=hashes)


def _pair_key(conflict) -> tuple[int, int]:
    return tuple(sorted((int(conflict.plan_a), int(conflict.plan_b))))


def _hub_metrics(graph, conflicts) -> dict[str, float]:
    degrees = dict(graph.degree())
    degree_values = np.asarray(list(degrees.values()), dtype=float)
    involvement = Counter()
    for conflict in conflicts:
        involvement.update((conflict.plan_a, conflict.plan_b))
    ranked_degree = sorted(graph.nodes, key=lambda node: (-degrees[node], node))
    ranked_involvement = sorted(graph.nodes, key=lambda node: (-involvement[node], node))
    result: dict[str, float] = {}
    for k in (1, 3, 5, 10):
        result[f"top{k}_degree_mean"] = float(np.mean([degrees[node] for node in ranked_degree[:k]])) if ranked_degree else 0.0
        result[f"top{k}_conflict_point_involvement"] = float(sum(involvement[node] for node in ranked_involvement[:k]))
    mean = float(degree_values.mean()) if len(degree_values) else 0.0
    result["degree_cv"] = float(degree_values.std() / mean) if mean else 0.0
    result["degree_max_to_mean"] = float(degree_values.max() / mean) if mean else 0.0
    return result


def score_row(row: dict) -> float:
    return (abs(row["det_Nc"] - 53) / 53 + abs(row["unc_Nc"] - 97) / 97
            + abs(row["CI_top10_point_coverage"] - TARGET_COVERAGE) / TARGET_COVERAGE)


def evaluate_traffic_seed(cfg: dict, grid, risk, hashes: dict[str, str], traffic_seed: int, scan_run_id: str) -> dict:
    started = time.perf_counter()
    row = dict(traffic_seed=int(traffic_seed), environment_seed=int(grid.seed), **hashes,
               error="", generation_failures=0, astar_failures=0, runtime_sec=0.0)
    try:
        tasks = sample_paper_random_traffic(grid, cfg, n_flights=100, seed=traffic_seed)
        plans = plan_paper_random_traffic(grid, risk, cfg, tasks)
        uncertain = detect_conflicts(plans, cfg, uncertain=True)
        deterministic = detect_conflicts(plans, cfg, uncertain=False)
        diag, _ = analyze_initial_conflict_network(plans, uncertain, deterministic, traffic_seed, scan_run_id, grid=grid)
        graph = build_conflict_network(plans, uncertain)
        row.update(
            det_Nc=len(deterministic), unc_Nc=len(uncertain),
            conflict_edges=graph.number_of_edges(), CI_top10_point_coverage=diag["top10_ci_point_coverage"],
            CI_top10_pair_coverage=diag["top10_ci_pair_coverage"],
            continuous_conflict_ratio=diag["continuous_conflict_point_ratio"],
            route_density_gini=diag["route_density_gini"],
            route_cell_reuse_p90=diag["route_cell_reuse_p90"], route_cell_reuse_max=diag["route_cell_reuse_max"],
            degree_gini=diag["degree_gini"], max_degree=diag["max_degree"],
            top10_degree_point_coverage=diag["top10_degree_point_coverage"],
            top10_conflict_point_share=diag["top10_conflict_point_share"],
            **_hub_metrics(graph, uncertain),
        )
        row["calibration_score"] = score_row(row)
        row["hub_rank_score"] = row["CI_top10_point_coverage"]
        level_a = 45 <= row["det_Nc"] <= 65 and 85 <= row["unc_Nc"] <= 115 and row["CI_top10_point_coverage"] >= 0.75
        level_b = 30 <= row["det_Nc"] <= 80 and 60 <= row["unc_Nc"] <= 130 and row["CI_top10_point_coverage"] >= 0.60
        row["candidate_tag"] = "A" if level_a else "B" if level_b else ""
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        row.update(error=message, generation_failures=1, astar_failures=int("A*" in message or "path" in message.lower()))
        for column in METRIC_COLUMNS:
            row.setdefault(column, np.nan)
        row.update(calibration_score=np.nan, hub_rank_score=np.nan, candidate_tag="")
    row["runtime_sec"] = time.perf_counter() - started
    return row


def _worker_evaluate(traffic_seed: int, scan_run_id: str) -> dict:
    return evaluate_traffic_seed(_WORKER["cfg"], _WORKER["grid"], _WORKER["risk"], _WORKER["hashes"], traffic_seed, scan_run_id)


METRIC_COLUMNS = [
    "det_Nc", "unc_Nc", "conflict_edges", "CI_top10_point_coverage", "CI_top10_pair_coverage",
    "continuous_conflict_ratio", "route_density_gini", "route_cell_reuse_p90", "route_cell_reuse_max",
    "degree_gini", "max_degree", "top10_degree_point_coverage", "top10_conflict_point_share",
    "top1_degree_mean", "top3_degree_mean", "top5_degree_mean", "top10_degree_mean",
    "top1_conflict_point_involvement", "top3_conflict_point_involvement", "top5_conflict_point_involvement", "top10_conflict_point_involvement",
    "degree_cv", "degree_max_to_mean",
]


def run_seed_batch(cfg: dict, environment_seed: int, seeds: list[int], jobs: int, scan_run_id: str):
    grid, risk, hashes = prepare_environment(cfg, environment_seed)
    if jobs == 1:
        return [evaluate_traffic_seed(cfg, grid, risk, hashes, seed, scan_run_id) for seed in seeds], hashes
    rows = []
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_worker, initargs=(cfg, environment_seed, hashes)) as pool:
        futures = {pool.submit(_worker_evaluate, seed, scan_run_id): seed for seed in seeds}
        for future in as_completed(futures):
            rows.append(future.result())
    return rows, hashes


def _write_rows(rows: list[dict], output: Path) -> None:
    pd.DataFrame(sorted(rows, key=lambda row: row["traffic_seed"])).to_csv(output, index=False)


def _candidate_details(directory: Path, cfg: dict, grid, risk, seed: int, scan_run_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=False)
    tasks = sample_paper_random_traffic(grid, cfg, n_flights=100, seed=seed)
    plans = plan_paper_random_traffic(grid, risk, cfg, tasks)
    uncertain = detect_conflicts(plans, cfg, uncertain=True)
    deterministic = detect_conflicts(plans, cfg, uncertain=False)
    write_conflicts_csv(uncertain, directory / "initial_conflicts_uncertain.csv")
    write_conflicts_csv(deterministic, directory / "initial_conflicts_deterministic.csv")
    diag, _ = analyze_initial_conflict_network(plans, uncertain, deterministic, seed, scan_run_id, directory, grid)
    graph = build_conflict_network(plans, uncertain)
    ci = collective_influence(graph, 2)
    degree = dict(graph.degree())
    point_involvement, pair_involvement = Counter(), Counter()
    for conflict in uncertain:
        point_involvement.update((conflict.plan_a, conflict.plan_b))
        pair_involvement.update((conflict.plan_a, conflict.plan_b))
    top10 = sorted(graph.nodes, key=lambda node: (-ci[node], node))[:10]
    union_points = sum(1 for conflict in uncertain if conflict.plan_a in top10 or conflict.plan_b in top10)
    pd.DataFrame([dict(run_id=scan_run_id, traffic_seed=seed, rank=rank, flight_id=node,
                       collective_influence=ci[node], degree=degree[node],
                       conflict_point_involvement=point_involvement[node], conflict_pair_involvement=pair_involvement[node],
                       top10_union_conflict_points=union_points)
                  for rank, node in enumerate(top10, 1)]).to_csv(directory / "top10_key_flights.csv", index=False)
    pairs = Counter(_pair_key(conflict) for conflict in uncertain)
    pd.DataFrame([dict(run_id=scan_run_id, traffic_seed=seed, plan_a=pair[0], plan_b=pair[1], conflict_points=count)
                  for pair, count in sorted(pairs.items())]).to_csv(directory / "conflict_pair_multiplicity.csv", index=False)
    reuse = Counter(cell for plan in plans for cell in set(plan.path))
    pd.DataFrame([dict(run_id=scan_run_id, traffic_seed=seed, grid_x=cell[0], grid_y=cell[1], grid_z=cell[2], distinct_flight_count=count)
                  for cell, count in sorted(reuse.items())]).to_csv(directory / "route_usage_summary.csv", index=False)
    pd.DataFrame([dict(run_id=scan_run_id, traffic_seed=seed, **record) for record in traffic_records(tasks)]).to_csv(directory / "traffic_tasks.csv", index=False)
    tag_run_csvs(directory, scan_run_id)


def _summary(values: pd.Series) -> dict[str, float]:
    return {"min": float(values.min()), "mean": float(values.mean()), "median": float(values.median()),
            "p90": float(values.quantile(.90)), "p95": float(values.quantile(.95)), "max": float(values.max())}


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    """Avoid a runtime dependency on the optional tabulate package."""
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for _, row in frame[columns].iterrows():
        body.append("| " + " | ".join("" if pd.isna(row[column]) else str(row[column]) for column in columns) + " |")
    return "\n".join([header, divider, *body])


def write_report(output: Path, rows: pd.DataFrame, scan_run_id: str, runtime: float) -> None:
    ok = rows[rows["error"].fillna("").eq("")].copy()
    coverage_stats = _summary(ok["CI_top10_point_coverage"]) if len(ok) else {}
    best_cal = ok.sort_values(["calibration_score", "traffic_seed"]).head(20)
    best_cov = ok.sort_values(["hub_rank_score", "traffic_seed"], ascending=[False, True]).head(20)
    report = ["# Traffic Seed Scan Report", "", f"Scan ID: `{scan_run_id}`", "",
              f"Completed seeds: {len(ok)}/{len(rows)}; failures: {len(rows)-len(ok)}; wall time: {runtime:.2f}s.", "",
              "## CI Top10 Point Coverage", "",
              json.dumps(coverage_stats, indent=2), "",
              f"Counts: >=0.60: {(ok['CI_top10_point_coverage'] >= .60).sum()}, >=0.70: {(ok['CI_top10_point_coverage'] >= .70).sum()}, >=0.80: {(ok['CI_top10_point_coverage'] >= .80).sum()}.", "",
              "## Best Calibration Candidates", "", _markdown_table(best_cal, ["traffic_seed", "det_Nc", "unc_Nc", "CI_top10_point_coverage", "calibration_score", "candidate_tag"]), "",
              "## Highest Coverage Candidates", "", _markdown_table(best_cov, ["traffic_seed", "det_Nc", "unc_Nc", "CI_top10_point_coverage", "hub_rank_score", "candidate_tag"]), ""]
    if coverage_stats and coverage_stats["max"] < .80:
        report += ["traffic seed alone is insufficient to explain the structural gap.", ""]
    high_outside = best_cov[(~best_cov["det_Nc"].between(45, 65)) | (~best_cov["unc_Nc"].between(85, 115))]
    if len(high_outside):
        report += ["high CI coverage alone does not reproduce the complete paper scenario.", ""]
    report += ["## Scope", "", "Only fixed-environment traffic sampling, A*, initial conflict detection, and CI analysis were executed. No FATA, ADM, Stage1, Stage2, or scheduler output was created."]
    (output / "traffic_seed_scan_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment-seed", type=int, default=2025)
    parser.add_argument("--traffic-seed-start", type=int, default=0)
    parser.add_argument("--traffic-seed-end", type=int, default=499)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output-root", default="outputs/traffic_seed_scans")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.traffic_seed_end < args.traffic_seed_start or args.n_jobs < 1:
        raise ValueError("seed range must be increasing and n-jobs must be positive")
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config, {"optimization": {"scheduler_mode": "paper_strict"}})
    environment_seed, _ = resolve_scene_seeds(cfg, environment_seed=args.environment_seed, traffic_seed=args.traffic_seed_start)
    validate_baseline(cfg)
    commit, dirty = git_metadata(root.parent)
    run_id = scan_id(environment_seed, args.traffic_seed_start, args.traffic_seed_end, commit)
    output = root / args.output_root / run_id
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    seeds = list(range(args.traffic_seed_start, args.traffic_seed_end + 1))
    grid, risk, hashes = prepare_environment(cfg, environment_seed)
    manifest = dict(scan_id=run_id, status="running", start_time=datetime.now().astimezone().isoformat(),
                    environment_seed=environment_seed, traffic_seed_start=args.traffic_seed_start, traffic_seed_end=args.traffic_seed_end,
                    n_jobs=args.n_jobs, git_commit=commit, git_dirty=dirty, python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(),
                    blas_threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
                    environment_initialization="parent once; each Windows worker regenerates once and asserts the same hashes", hashes=hashes,
                    population_model=cfg["population_model"], population=cfg["population"], astar_distance_scale_mode=cfg["astar_distance_scale_mode"],
                    paper_scene=cfg["paper_scene"], conflict=cfg["conflict"], network=cfg["network"], optimizers_executed=False)
    (output / "scan_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    rows: list[dict] = []
    if args.n_jobs == 1:
        for index, seed in enumerate(seeds, 1):
            rows.append(evaluate_traffic_seed(cfg, grid, risk, hashes, seed, run_id))
            print(f"[{index}/{len(seeds)}] traffic_seed={seed}", flush=True)
            if index % 50 == 0:
                _write_rows(rows, output / "traffic_seed_scan.csv")
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs, initializer=_init_worker, initargs=(cfg, environment_seed, hashes)) as pool:
            futures = {pool.submit(_worker_evaluate, seed, run_id): seed for seed in seeds}
            for index, future in enumerate(as_completed(futures), 1):
                seed = futures[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    rows.append(dict(traffic_seed=seed, environment_seed=environment_seed, **hashes, error=f"WorkerError: {exc}", generation_failures=1, astar_failures=0, runtime_sec=np.nan))
                print(f"[{index}/{len(seeds)}] traffic_seed={seed}", flush=True)
                if index % 50 == 0:
                    _write_rows(rows, output / "traffic_seed_scan.csv")
    _write_rows(rows, output / "traffic_seed_scan.csv")
    frame = pd.read_csv(output / "traffic_seed_scan.csv")
    ok = frame[frame["error"].fillna("").eq("")]
    calibration = ok.sort_values(["calibration_score", "traffic_seed"]).head(20)
    coverage_rank = ok.sort_values(["hub_rank_score", "traffic_seed"], ascending=[False, True]).head(20)
    calibration.to_csv(output / "top20_calibration.csv", index=False)
    coverage_rank.to_csv(output / "top20_coverage.csv", index=False)
    candidate_seeds = sorted(set(calibration["traffic_seed"].astype(int)) | set(coverage_rank["traffic_seed"].astype(int)))
    for seed in candidate_seeds:
        _candidate_details(output / "candidates" / f"traffic_seed_{seed}", cfg, grid, risk, seed, run_id)
    for label, table in (("calibration", calibration.head(5)), ("coverage", coverage_rank.head(5))):
        figures = output / "network_figures" / label
        figures.mkdir(parents=True, exist_ok=True)
        for seed in table["traffic_seed"].astype(int):
            tasks = sample_paper_random_traffic(grid, cfg, n_flights=100, seed=seed)
            plans = plan_paper_random_traffic(grid, risk, cfg, tasks)
            conflicts = detect_conflicts(plans, cfg, uncertain=True)
            graph = build_conflict_network(plans, conflicts)
            metrics = pd.DataFrame([dict(flight_id=node, collective_influence=value) for node, value in collective_influence(graph, 2).items()])
            plot_network(graph, metrics, figures / f"conflict_network_seed_{seed}.png")
    runtime = time.perf_counter() - started
    manifest.update(status="completed", end_time=datetime.now().astimezone().isoformat(), total_runtime_sec=runtime,
                    completed=len(ok), failures=len(frame) - len(ok), candidate_detail_seeds=candidate_seeds)
    (output / "scan_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(output, frame, run_id, runtime)
    best = calibration.iloc[0] if len(calibration) else None
    high = coverage_rank.iloc[0] if len(coverage_rank) else None
    print(f"best calibration seed: {int(best.traffic_seed) if best is not None else 'none'}", flush=True)
    print(f"highest coverage seed: {int(high.traffic_seed) if high is not None else 'none'}", flush=True)
    print(f"max CI Top10 point coverage: {float(high.CI_top10_point_coverage) if high is not None else float('nan'):.6f}", flush=True)
    print(f"output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
