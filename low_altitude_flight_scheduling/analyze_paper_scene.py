"""Generate one initial paper scene and report its network, without optimization."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import pandas as pd

from src.config import load_config
from src.conflict_detection import detect_conflicts
from src.run_archive import generate_run_id, git_metadata
from src.scene_diagnostics import analyze_initial_conflict_network, generate_and_analyze_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--outputs", default="outputs/initial_network_runs")
    parser.add_argument("--baseline-plans", default="outputs/initial_plans.pkl")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cfg = load_config(root / args.config, {"optimization": {"scheduler_mode": "paper_strict"}})
    cfg["scene_mode"] = "paper_strict_random"
    commit, dirty = git_metadata(root.parent)
    run_id = generate_run_id("paper_strict_random", "initial_network", args.seed, commit)
    out = root / args.outputs / run_id
    out.mkdir(parents=True, exist_ok=False)
    row, sensitivity = generate_and_analyze_scene(cfg, args.seed, run_id, out)
    baseline_path = root / args.baseline_plans
    baseline = None
    if baseline_path.exists():
        with baseline_path.open("rb") as fh:
            plans = pickle.load(fh)
        if any(p.generation_mode == "paper_random" for p in plans):
            raise ValueError("Baseline must contain the existing engineering scene, not paper_random")
        baseline, _ = analyze_initial_conflict_network(plans, detect_conflicts(plans, cfg, True),
                                                       detect_conflicts(plans, cfg, False), args.seed,
                                                       "archived_heterogeneous_recount", out / "heterogeneous_baseline")
    metrics = [
        ("uncertain_conflict_points", "conflict_points_uncertain", 97),
        ("deterministic_conflict_points", "conflict_points_deterministic", 53),
        ("CI_top10_point_coverage", "top10_ci_point_coverage", 78 / 97),
        ("CI_top10_pair_coverage", "top10_ci_pair_coverage", None),
        ("network_edges", "network_edges", None),
        ("degree_gini", "degree_gini", None),
        ("unique_pair_cell_conflicts", "unique_spatial_conflict_points", None),
    ]
    pd.DataFrame([dict(metric=name, current_heterogeneous=baseline[key] if baseline else None,
                       paper_random=row[key], paper_reference=reference) for name, key, reference in metrics]).to_csv(
                           out / "network_reproduction_comparison.csv", index=False)
    print(f"Deterministic conflict points: {row['conflict_points_deterministic']}")
    print(f"Uncertain conflict points: {row['conflict_points_uncertain']}")
    print(f"Unique pair-cell conflicts: {row['unique_spatial_conflict_points']}")
    print(f"Network edges: {row['network_edges']}")
    print(f"Max degree: {row['max_degree']}")
    print(f"Degree Gini: {row['degree_gini']:.6f}")
    for item in sensitivity:
        print(f"CI l={item['ci_l']} Top10 coverage: {item['top10_point_coverage']:.6f}")
    print("Paper references: deterministic=53; uncertain=97; CI Top10 coverage=78/97=0.8041")
    (out / "scene_manifest.json").write_text(json.dumps(dict(
        run_id=run_id, seed=args.seed, scene_mode="paper_strict_random", git_commit=commit,
        git_dirty=dirty, optimizers_executed=False, config=cfg,
        baseline_path=str(baseline_path) if baseline else None,
        baseline_note="Existing engineering paths recounted with the same 30s and unchanged sigma as paper_random; no baseline regeneration.",
    ), indent=2), encoding="utf-8")
    baseline_text = (f"Archived heterogeneous baseline, recounted at 30s: deterministic={baseline['conflict_points_deterministic']}, "
                     f"uncertain={baseline['conflict_points_uncertain']}, CI point coverage={baseline['top10_ci_point_coverage']:.4f}."
                     if baseline else "No archived heterogeneous baseline was available; comparison entries are blank.")
    report = f"""# Initial Network Reproduction Report

Run: {run_id}; seed: {args.seed}; scene: paper_strict_random. No FATA, ADM, Stage1 or Stage2 was executed.

1. The heterogeneous scene used OD hotspots and mixed OD patterns, peaked takeoff times, variable speeds, forced cruise altitude and optional corridors. These introduce structure absent from the published random initial-scene description. {baseline_text}

2. paper_random restores random ground OD sampling, 100 plans, distance near 6km, uniform ETD in [0,1800] seconds, constant 10m/s, A* risk/distance weights 0.8/0.2 and t_conflict=30s, alpha=0.05. No forced corridors, waypoints, altitude preference, ground penalty or route bias is used. The A* heuristic and path distance use meters; the engineering distance division by 100 is disabled only at this initial generation call. The 1200m tolerance is an implementation assumption. Random OD positions are discrete ground cell centers, conditioned on feasibility and the fixed distance band; this discretization is an implementation assumption. Random city/population maps and sigma0={cfg['conflict']['sigma0']}, sigma_rate={cfg['conflict']['sigma_rate']} remain implementation assumptions, unchanged in this round.

3. CI uses a binary simple graph with one node per plan and one edge per potentially conflicting pair. CI=(k_u-1)*sum(k_v-1) on the exact l-hop boundary. Edge count metadata does not enter CI. Top10 uses ordinary CI with flight-id tie breaking. This follows the published formula; weighted/adaptive key selection is not used for this scene.

4. l=2 is an implementation assumption, not a disclosed paper parameter. Radius sensitivity is diagnostic only; default selection remains l=2.

5. Raw events={row['raw_conflict_events']}; unique pair-cell points={row['unique_spatial_conflict_points']}; duplicate extra index events={row['same_pair_same_cell_multiple_index_count']}. The main Conflict definition is unchanged. A repeated pair-cell may correspond to multiple temporal events, so the paper counting convention needs manual review when these counts differ. Continuous conflict ratio={row['continuous_conflict_point_ratio']:.4f}; continuity uses the existing adjacency definition and is also a diagnostic convention. Top1/5/10 contribution shares count the union of events incident to the selected flights, avoiding double counting.

6. seed2025 (or the explicitly requested seed {args.seed}) CI l=2 Top10 point coverage={row['top10_ci_point_coverage']:.6f}, pair coverage={row['top10_ci_pair_coverage']:.6f}. Paper point coverage=78/97=0.804124. Deterministic={row['conflict_points_deterministic']} vs 53; uncertain={row['conflict_points_uncertain']} vs 97.

7. Scene and path-generation assumptions change the underlying network. Counting and radius effects are quantified below; there is no evidence here that changing the ordinary CI implementation is warranted. Aggregate outputs alone cannot isolate the effects of undisclosed city maps, population maps, seed and sigma. Comparing only the total conflict count is insufficient to claim structural reproduction.

```json
{json.dumps(sensitivity, indent=2)}
```

8. Calibration is optional when a scene resembling published initial statistics is needed, and must be labelled calibrated reproduction scene. A small scan can reveal variability, but does not establish that the paper scene can be recovered. The unbiased seed={args.seed} scene remains the strict result. Calibration must never use final optimization results and must not change CI, key count, tolerance or sigma to fit the targets.
"""
    (out / "initial_network_reproduction_report.md").write_text(report, encoding="utf-8")
    for name in ("initial_network_reproduction_report.md", "network_reproduction_comparison.csv"):
        (root / "outputs" / name).write_bytes((out / name).read_bytes())
    print(f"outputs: {out}")


if __name__ == "__main__":
    main()
