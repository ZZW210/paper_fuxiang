"""Deprecated generator for the rejected env2025 / traffic316 plan scene.

Its existing outputs remain as historical evidence only. Formal reproduction
runs use data/baseline_initial_plans.pkl and must not regenerate this scene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from src.config import load_config, resolve_scene_seeds
from src.conflict_detection import detect_conflicts
from src.conflict_network import build_conflict_network
from src.flight_plan import plan_paper_random_traffic, sample_paper_random_traffic
from src.grid import AirspaceGrid
from src.risk_map import generate_risk_map
from src.visualization import write_fata_3d_html


EXPECTED_MULTI_KEYS = [73, 47, 49, 56, 33, 78, 42, 25, 1, 63]
FOCUS_IDS = [49, 73, 98, 47, 56]


def _ci_l2(topology: nx.Graph, values: dict[int, int]) -> dict[int, float]:
    return {
        node: float(
            (values[node] - 1)
            * sum(
                max(0, values[other] - 1)
                for other, distance in nx.single_source_shortest_path_length(topology, node, cutoff=2).items()
                if distance == 2
            )
        )
        for node in topology.nodes
    }


def _node_metrics(plans, conflicts) -> pd.DataFrame:
    graph = build_conflict_network(plans, conflicts)
    binary_degree = dict(graph.degree())
    multi_degree = {plan.id: 0 for plan in plans}
    for conflict in conflicts:
        multi_degree[conflict.flight_a] += 1
        multi_degree[conflict.flight_b] += 1
    ci_multi = _ci_l2(graph, multi_degree)
    rows = [
        {
            "flight_id": plan.id,
            "conflict_point_involvement": multi_degree[plan.id],
            "unique_conflict_partners": binary_degree[plan.id],
            "multi_degree": multi_degree[plan.id],
            "CI_multi_l2": ci_multi[plan.id],
        }
        for plan in plans
    ]
    frame = pd.DataFrame(rows)
    frame["multi_rank"] = frame["CI_multi_l2"].rank(method="first", ascending=False).astype(int)
    return frame.sort_values("multi_rank").reset_index(drop=True)


def _dense_sidebar(metrics: pd.DataFrame) -> str:
    lines = [
        "<b>Top15 dense-conflict flights</b>",
        "<span style='color:#666'>id | points | partners | D_multi | CI_l2 | rank</span>",
    ]
    for row in metrics.head(15).itertuples(index=False):
        lines.append(
            f"{row.flight_id} | {row.conflict_point_involvement} | "
            f"{row.unique_conflict_partners} | {row.multi_degree} | "
            f"{row.CI_multi_l2:.0f} | {row.multi_rank}"
        )
    lines.append("<br><b>Quick-focus:</b> 49, 73, 98, 47, 56")
    return "<br>".join(lines)


def main() -> None:
    raise RuntimeError(
        "deprecated_bad_flight_plan_scene: this generator is not a formal baseline. "
        "Use data/baseline_initial_plans.pkl through the baseline diagnostics instead."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-seed", type=int, default=2025)
    parser.add_argument("--traffic-seed", type=int, default=316)
    args = parser.parse_args()
    if (args.environment_seed, args.traffic_seed) != (2025, 316):
        raise ValueError("This visualization is intentionally pinned to env2025 / traffic316.")

    root = Path(__file__).resolve().parent
    cfg = load_config(root / "config.yaml", {"optimization": {"scheduler_mode": "paper_strict"}})
    resolve_scene_seeds(cfg, environment_seed=args.environment_seed, traffic_seed=args.traffic_seed)
    grid = AirspaceGrid.from_config(cfg, seed=args.environment_seed)
    risk_map = generate_risk_map(grid, cfg)
    traffic = sample_paper_random_traffic(grid, cfg, 100, args.traffic_seed)
    plans = plan_paper_random_traffic(grid, risk_map, cfg, traffic)
    deterministic = detect_conflicts(plans, cfg, uncertain=False)
    uncertain = detect_conflicts(plans, cfg, uncertain=True)
    if (len(deterministic), len(uncertain)) != (53, 107):
        raise RuntimeError(
            f"Fixed scene validation failed: expected deterministic/uncertain 53/107, "
            f"got {len(deterministic)}/{len(uncertain)}."
        )

    metrics = _node_metrics(plans, uncertain)
    observed_keys = metrics.head(10)["flight_id"].astype(int).tolist()
    if observed_keys != EXPECTED_MULTI_KEYS:
        raise RuntimeError(f"Multi-degree CI l=2 Top10 mismatch: {observed_keys}")
    dense_ids = metrics.head(15)["flight_id"].astype(int).tolist()
    output = root / "outputs" / "new_scene_3d_env2025_traffic316"
    output.mkdir(parents=True, exist_ok=True)

    write_fata_3d_html(
        grid, plans, uncertain, output / "fata_3d_before.html",
        "New scene before scheduling (env2025 / traffic316)", highlight_ids=FOCUS_IDS,
    )
    write_fata_3d_html(
        grid, plans, uncertain, output / "fata_3d_before_all.html",
        "New scene before scheduling: all 100 flights", highlight_ids=FOCUS_IDS,
    )
    write_fata_3d_html(
        grid, plans, uncertain, output / "fata_3d_before_multi_key.html",
        "New scene before scheduling: multi-degree CI l=2 Top10", key_ids=EXPECTED_MULTI_KEYS,
        highlight_ids=FOCUS_IDS,
    )
    write_fata_3d_html(
        grid, plans, uncertain, output / "fata_3d_before_dense_only.html",
        "New scene before scheduling: dense-conflict Top15", key_ids=dense_ids,
        highlight_ids=FOCUS_IDS, dim_other_routes=True, sidebar_html=_dense_sidebar(metrics),
    )
    metrics.to_csv(output / "dense_conflict_flight_metrics.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({
        "environment_seed": 2025,
        "traffic_seed": 316,
        "deterministic_conflicts": len(deterministic),
        "uncertain_conflicts": len(uncertain),
        "multi_degree_ci_l2_top10": EXPECTED_MULTI_KEYS,
        "dense_conflict_top15": dense_ids,
        "focus_flights": FOCUS_IDS,
        "stages_executed": [],
    }, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
