"""Initial scene diagnostics; no scheduler or optimizer is executed."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import pickle
import time

import networkx as nx
import numpy as np
import pandas as pd

from .conflict_detection import detect_conflicts, group_continuous_conflicts
from .conflict_network import build_conflict_network, collective_influence
from .flight_plan import generate_flight_plans
from .grid import AirspaceGrid
from .risk_map import generate_risk_map


CALIBRATION_NOTE = (
    "This scene was selected by calibration against published aggregate initial statistics "
    "because the original random seed and full simulation environment were not disclosed."
)


def degree_gini(values):
    values = np.sort(np.asarray(values, dtype=float))
    if not len(values) or not values.sum():
        return 0.0
    return float(2 * np.dot(np.arange(1, len(values) + 1), values) / (len(values) * values.sum())
                 - (len(values) + 1) / len(values))


def pair_key(c):
    return tuple(sorted((c.plan_a, c.plan_b)))


def coverage(conflicts, ids):
    ids = set(ids)
    covered = [c for c in conflicts if c.plan_a in ids or c.plan_b in ids]
    pairs = {pair_key(c) for c in conflicts}
    return len(covered) / len(conflicts) if conflicts else 0.0, len({pair_key(c) for c in covered}) / len(pairs) if pairs else 0.0


def ci_top10(graph, radius):
    scores = collective_influence(graph, radius)
    return sorted(graph.nodes, key=lambda pid: (-scores[pid], pid))[:10]


def analyze_initial_conflict_network(plans, conflicts, deterministic, seed, run_id, output_dir=None):
    graph = build_conflict_network(plans, conflicts)
    degrees = dict(graph.degree())
    degree_values = np.asarray(list(degrees.values()), dtype=float)
    counts = Counter(pair_key(c) for c in conflicts)
    spatial = Counter((*pair_key(c), tuple(c.cell)) for c in conflicts)
    contribution = Counter()
    for c in conflicts:
        contribution.update((c.plan_a, c.plan_b))
    ranked_contribution = sorted(graph.nodes, key=lambda pid: (-contribution[pid], pid))
    degree_ids = sorted(graph.nodes, key=lambda pid: (-degrees[pid], pid))[:10]
    key_ids = ci_top10(graph, 2)
    point_cov, pair_cov = coverage(conflicts, key_ids)
    segments = [s for s in group_continuous_conflicts(conflicts) if len(s.conflicts) > 1]
    continuous_points = sum(len(s.conflicts) for s in segments)
    lcc = max((len(c) for c in nx.connected_components(graph)), default=0)
    edge_counts = list(counts.values())
    row = dict(
        run_id=run_id, seed=seed,
        conflict_points_uncertain=len(conflicts), conflict_points_deterministic=len(deterministic),
        conflict_pairs=len(counts), network_edges=graph.number_of_edges(),
        mean_conflict_points_per_edge=float(np.mean(edge_counts)) if edge_counts else 0.0,
        median_conflict_points_per_edge=float(np.median(edge_counts)) if edge_counts else 0.0,
        max_conflict_points_per_edge=max(edge_counts, default=0),
        active_conflict_flights=int(sum(v > 0 for v in degree_values)), isolated_flights=int(sum(v == 0 for v in degree_values)),
        mean_degree=float(degree_values.mean()) if len(degree_values) else 0.0,
        max_degree=max(degrees.values(), default=0), degree_std=float(degree_values.std()) if len(degree_values) else 0.0,
        degree_gini=degree_gini(degree_values), largest_connected_component_size=lcc,
        largest_connected_component_ratio=lcc / len(plans) if plans else 0.0,
        top10_ci_point_coverage=point_cov, top10_ci_pair_coverage=pair_cov,
        top10_degree_point_coverage=coverage(conflicts, degree_ids)[0],
        top1_conflict_point_share=coverage(conflicts, ranked_contribution[:1])[0],
        top5_conflict_point_share=coverage(conflicts, ranked_contribution[:5])[0],
        top10_conflict_point_share=coverage(conflicts, ranked_contribution[:10])[0],
        continuous_conflict_segments=len(segments), continuous_conflict_points=continuous_points,
        continuous_conflict_point_ratio=continuous_points / len(conflicts) if conflicts else 0.0,
        raw_conflict_events=len(conflicts), unique_spatial_conflict_points=len(spatial),
        same_pair_same_cell_multiple_index_count=sum(v - 1 for v in spatial.values()),
        same_pair_same_cell_multiple_index_groups=sum(v > 1 for v in spatial.values()),
    )
    sensitivity = []
    for radius in (1, 2, 3):
        ids = ci_top10(graph, radius)
        pc, ec = coverage(conflicts, ids)
        reduced = graph.copy()
        reduced.remove_nodes_from(ids)
        sensitivity.append(dict(ci_l=radius, top10_flight_ids=json.dumps(ids), top10_point_coverage=pc,
                                top10_pair_coverage=ec, remaining_edges_after_top10_removal=reduced.number_of_edges(),
                                lcc_ratio_after_top10_removal=max((len(c) for c in nx.connected_components(reduced)), default=0) / len(plans) if plans else 0.0))
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([row]).to_csv(out / "initial_network_diagnostics.csv", index=False)
        pd.DataFrame(sensitivity).to_csv(out / "ci_radius_sensitivity.csv", index=False)
        pd.DataFrame([dict(flight_id=pid, degree=degrees[pid], collective_influence=collective_influence(graph, 2)[pid],
                           conflict_point_involvement=contribution[pid], is_ci_top10=pid in key_ids) for pid in sorted(graph.nodes)]).to_csv(out / "initial_node_diagnostics.csv", index=False)
    return row, sensitivity


def calibration_score(row):
    return (abs(row["conflict_points_deterministic"] - 53) / 53
            + abs(row["conflict_points_uncertain"] - 97) / 97
            + abs(row["top10_ci_point_coverage"] - 78 / 97))


def write_calibration_report(directory, rows, seed_start, seed_end):
    if not rows:
        raise ValueError("Calibration requires completed initial scenes")
    best = min(rows, key=lambda row: (calibration_score(row), row["seed"]))
    report = (f"# Calibrated Reproduction Scene\n\n{CALIBRATION_NOTE}\n\n"
              f"Seeds: {seed_start}..{seed_end} inclusive; candidates: {len(rows)}.\n\n"
              "Targets: deterministic=53, uncertain=97, CI Top10 point coverage=78/97.\n\n"
              "Score = abs(det-53)/53 + abs(unc-97)/97 + abs(coverage-78/97).\n\n"
              f"Best seed: {best['seed']}; score: {calibration_score(best):.6f}.\n\n"
              f"```json\n{json.dumps(best, indent=2)}\n```\n\n"
              "Only published aggregate initial statistics were used. This is not the exact original scene.\n")
    (Path(directory) / "scene_calibration_report.md").write_text(report, encoding="utf-8")
    return best


def generate_and_analyze_scene(cfg, seed, run_id, output_dir=None):
    start = time.perf_counter()
    cfg = copy.deepcopy(cfg)
    cfg["flight"]["random_seed"] = seed
    grid = AirspaceGrid.from_config(cfg, seed=seed)
    risk = generate_risk_map(grid, cfg)
    plans = generate_flight_plans(grid, risk, cfg, seed=seed)
    uncertain = detect_conflicts(plans, cfg, uncertain=True)
    deterministic = detect_conflicts(plans, cfg, uncertain=False)
    row, sensitivity = analyze_initial_conflict_network(plans, uncertain, deterministic, seed, run_id, output_dir)
    row.update(runtime=time.perf_counter() - start, deterministic_Nc=len(deterministic), uncertain_Nc=len(uncertain),
               edges=row["network_edges"], CI_top10_point_coverage=row["top10_ci_point_coverage"],
               CI_top10_pair_coverage=row["top10_ci_pair_coverage"], continuous_ratio=row["continuous_conflict_point_ratio"],
               scene_mode=cfg.get("scene_mode", "paper_strict_random"))
    row["calibration_score"] = calibration_score(row)
    if output_dir is not None:
        out = Path(output_dir)
        pd.DataFrame([row]).to_csv(out / "initial_network_diagnostics.csv", index=False)
        from .conflict_detection import write_conflicts_csv
        write_conflicts_csv(uncertain, out / "initial_conflicts_uncertain.csv")
        write_conflicts_csv(deterministic, out / "initial_conflicts_deterministic.csv")
        np.save(out / "obstacles.npy", grid.obstacles)
        np.save(out / "risk_map.npy", risk)
        with (out / "initial_plans.pkl").open("wb") as fh:
            pickle.dump(plans, fh)
        (out / "scene_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return row, sensitivity
