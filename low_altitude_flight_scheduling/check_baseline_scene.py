"""Recount the immutable historical baseline with current paper-strict conflict settings."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_plans import BASELINE_SOURCE_COMMIT, baseline_metadata, load_baseline_plans
from src.config import load_config
from src.conflict_detection import count_conflict_pairs, detect_conflicts
from src.conflict_network import build_conflict_network, collective_influence
from src.flight_plan import plot_routes_3d
from src.grid import AirspaceGrid
from src.scene_diagnostics import coverage
from src.visualization import write_fata_3d_html


def main() -> None:
    root = Path(__file__).resolve().parent
    out = root / "outputs" / "old_baseline_scene_check"
    out.mkdir(parents=True, exist_ok=True)
    metadata = baseline_metadata(root)
    plans = load_baseline_plans(root)
    cfg = load_config(root / "config.yaml", {"optimization": {"scheduler_mode": "paper_strict"}})
    cfg["conflict"]["t_conflict"] = 30.0
    grid = AirspaceGrid.from_config(cfg, seed=cfg.get("environment_seed", 2025))
    deterministic = detect_conflicts(plans, cfg, uncertain=False)
    uncertain = detect_conflicts(plans, cfg, uncertain=True)
    graph = build_conflict_network(plans, uncertain)
    degrees = dict(graph.degree())
    ci = collective_influence(graph, l=int(cfg["network"]["ci_l"]))
    top10 = sorted(graph.nodes, key=lambda fid: (-ci[fid], fid))[:10]
    point_coverage, pair_coverage = coverage(uncertain, top10)

    levels = [cell[2] for plan in plans for cell in plan.path]
    non_endpoint_levels = [cell[2] for plan in plans for cell in plan.path[1:-1]]
    altitude_distribution = Counter(levels)
    node_rows = [
        dict(flight_id=plan.id, degree=degrees[plan.id], collective_influence=ci[plan.id],
             conflict_point_involvement=sum(c.plan_a == plan.id or c.plan_b == plan.id for c in uncertain),
             ci_rank=sorted(graph.nodes, key=lambda fid: (-ci[fid], fid)).index(plan.id) + 1)
        for plan in plans
    ]
    pd.DataFrame(node_rows).sort_values("ci_rank").to_csv(out / "baseline_node_metrics.csv", index=False)
    plot_routes_3d(grid, plans, out / "initial_routes.png", "Historical baseline initial routes")
    write_fata_3d_html(grid, plans, uncertain, out / "fata_3d_before.html", "Historical baseline before scheduling", top10)
    report = f"""# Historical Baseline Scene Check

No optimizer, Stage1, Stage2, FATA, seed scan, A* regeneration, OD sampling, or altitude reassignment was executed.

- Baseline source commit: `{BASELINE_SOURCE_COMMIT}`
- Baseline plan SHA256: `{metadata['sha256']}`
- Flight count: {len(plans)}
- Current paper-strict conflict settings: `t_conflict={cfg['conflict']['t_conflict']}`, `alpha={cfg['conflict']['alpha']}`, `sigma0={cfg['conflict']['sigma0']}`, `sigma_rate={cfg['conflict']['sigma_rate']}`
- Route altitude distribution (grid z): `{dict(sorted(altitude_distribution.items()))}`
- Route altitude mean/min/max (grid z): `{float(np.mean(levels)):.4f}` / `{min(levels)}` / `{max(levels)}`
- Route points at z=0: `{100 * sum(level == 0 for level in levels) / len(levels):.4f}%`
- Non-endpoint route points at z=0: `{100 * sum(level == 0 for level in non_endpoint_levels) / max(1, len(non_endpoint_levels)):.4f}%`
- Deterministic conflict points: {len(deterministic)}
- Uncertain conflict points: {len(uncertain)}
- Unique conflict pairs: {count_conflict_pairs(uncertain)}
- Mean/max binary degree: `{float(np.mean(list(degrees.values()))):.4f}` / `{max(degrees.values(), default=0)}`
- CI l={cfg['network']['ci_l']} Top10: `{top10}`
- Top10 conflict-point coverage: `{point_coverage:.6f}`
- Top10 conflict-pair coverage: `{pair_coverage:.6f}`
"""
    (out / "baseline_scene_report.md").write_text(report, encoding="utf-8")
    (out / "manifest.json").write_text(json.dumps({
        "baseline_source_commit": BASELINE_SOURCE_COMMIT,
        "baseline_plan_hash": metadata["sha256"],
        "number_of_flights": len(plans),
        "deterministic_conflicts": len(deterministic),
        "uncertain_conflicts": len(uncertain),
        "unique_conflict_pairs": count_conflict_pairs(uncertain),
        "max_degree": max(degrees.values(), default=0),
        "ci_top10": top10,
        "top10_conflict_point_coverage": point_coverage,
        "stages_executed": [],
    }, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
