"""Extended key-ratio sensitivity analysis; preserves the 3%--12% Table 9 range."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_key_ratio import (_validate_fixed_scene, initial_ci_ranking, key_ids_from_ranking,
                               run_one)
from src.config import load_config, resolve_scene_seeds
from src.conflict_detection import detect_conflicts
from src.flight_plan import plan_paper_random_traffic, sample_paper_random_traffic
from src.grid import AirspaceGrid
from src.risk_map import generate_risk_map

EXTENDED_RATIOS = [.13, .14, .15, .16, .17, .18, .20, .22, .25]
TABLE9_MAX_RATIO = .12


def build_extended_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the fixed three paired optimizer seeds for each ratio."""
    rows = []
    for ratio, group in raw.groupby("key_ratio", sort=True):
        rows.append(dict(
            key_ratio=float(ratio), K=int(group["K"].iloc[0]),
            key_conflict_point_coverage=float(group["key_conflict_point_coverage"].iloc[0]),
            stage1_remaining_conflicts_mean=float(group["stage1_remaining_conflicts"].mean()),
            stage1_remaining_conflicts_std=float(group["stage1_remaining_conflicts"].std(ddof=0)),
            final_remaining_conflicts_mean=float(group["remaining_conflicts"].mean()),
            final_remaining_conflicts_std=float(group["remaining_conflicts"].std(ddof=0)),
            zero_conflict_success_rate=float((group["remaining_conflicts"] == 0).mean()),
            delayed_flights_mean=float(group["delayed_flight_count"].mean()),
            total_changed_flights_mean=float(group["total_changed_flights"].mean()),
            final_fitness_mean=float(group["final_fitness"].mean()),
            risk_increase_mean_percent=float(group["risk_increase_percent"].mean()),
            stage1_dimension=float(group["stage1_dimension"].mean()),
            stage2_dimension=float(group["stage2_dimension"].mean()),
        ))
    return pd.DataFrame(rows).sort_values("key_ratio")


def _plot(table: pd.DataFrame, column: str, ylabel: str, title: str, path: Path, error: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(table["key_ratio"], table[column], marker="o", linewidth=1.8)
    if error:
        ax.fill_between(table["key_ratio"], table[column] - table[error], table[column] + table[error], alpha=.16)
    ax.axvspan(.03, TABLE9_MAX_RATIO, alpha=.08, color="#4c78a8", label="paper Table 9 range")
    ax.axvspan(.13, .25, alpha=.08, color="#f58518", label="extended sensitivity range")
    ax.set(xlabel="Key-flight ratio", ylabel=ylabel, title=title)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _best_ratio(table: pd.DataFrame) -> pd.Series:
    stable = table[table["zero_conflict_success_rate"].eq(1.0)]
    pool = stable if len(stable) else table
    return pool.sort_values(["final_remaining_conflicts_mean", "final_fitness_mean", "key_ratio"]).iloc[0]


def _report(path: Path, table: pd.DataFrame) -> None:
    best = _best_ratio(table)
    zero = table[table["zero_conflict_success_rate"].gt(0)]
    stable = table[table["zero_conflict_success_rate"].eq(1.0)]
    first_zero = f"{zero.iloc[0].key_ratio:.2f}" if len(zero) else "none in 3 repeats"
    first_stable = f"{stable.iloc[0].key_ratio:.2f}" if len(stable) else "none in 3 repeats"
    after_best = table[table["key_ratio"] > best.key_ratio]
    u_shape = bool(len(after_best) and after_best["final_fitness_mean"].min() > best.final_fitness_mean)
    text = f"""# Extended Key-Ratio Sensitivity Analysis

The 3%--12% rows are reused unchanged from the completed Table 9 reproduction debug experiment. Ratios above 0.12 are a **reproduction-specific extended sensitivity analysis**, not part of the paper's original Table 9 range. Every ratio uses the same fixed environment, traffic, initial CI ranking, FATA/ADM settings, and optimizer seeds 0, 1, and 2.

1. First ratio with a zero-conflict result: {first_zero}.
2. First ratio with all three runs at zero conflict: {first_stable}.
3. Best ratio under the stated ordering (lowest final mean conflicts, then fitness): {best.key_ratio:.2f} (K={int(best.K)}).
4. Best ratio {'is' if best.key_ratio > .10 else 'is not'} greater than the paper's 0.10.
5. Its initial CI conflict point coverage is {best.key_conflict_point_coverage:.6f}.
6. {'The fixed-scene result requires more key plans than 0.10, which is consistent with a different random network structure.' if best.key_ratio > .10 else 'The fixed-scene result does not establish that more key plans than the paper are required.'}
7. {'All larger tested ratios have higher mean fitness than the selected optimum, producing a local improve-then-worsen pattern.' if u_shape else 'The extended samples do not establish a strict U-shaped fitness pattern; inspect the plotted means rather than inferring one.'}

No algorithm parameter was changed to make a ratio win. CI coverage is diagnostic only and is not the selection criterion.
"""
    path.write_text(text, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-analysis", required=True, help="Completed 3%%--12%% debug directory")
    parser.add_argument("--environment-seed", type=int, default=2025)
    parser.add_argument("--traffic-seed", type=int, default=316)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeats != 3:
        raise ValueError("this extension is defined for the fixed debug repeats=3")
    root = Path(__file__).resolve().parent
    base = (root / args.base_analysis).resolve()
    base_raw = pd.read_csv(base / "raw_runs.csv")
    expected = {round(ratio, 2) for ratio in np.arange(.03, .13, .01)}
    if set(np.round(base_raw["key_ratio"].unique(), 2)) != expected or base_raw.groupby("key_ratio").size().ne(3).any():
        raise ValueError("base analysis must contain exactly the completed 3%--12% three-repeat records")
    cfg = load_config(root / "config.yaml", {"optimization": {"scheduler_mode": "paper_strict", "n_jobs": args.n_jobs}})
    resolve_scene_seeds(cfg, environment_seed=args.environment_seed, traffic_seed=args.traffic_seed)
    _validate_fixed_scene(cfg)
    grid = AirspaceGrid.from_config(cfg, seed=args.environment_seed)
    risk_map = generate_risk_map(grid, cfg)
    tasks = sample_paper_random_traffic(grid, cfg, 100, args.traffic_seed)
    plans = plan_paper_random_traffic(grid, risk_map, cfg, tasks)
    deterministic = detect_conflicts(plans, cfg, uncertain=False)
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    if (len(deterministic), len(conflicts)) != (53, 107):
        raise RuntimeError(f"fixed scene is not reproducible: deterministic={len(deterministic)}, uncertain={len(conflicts)}")
    ranking = initial_ci_ranking(plans, conflicts)
    out = base / "extended_sensitivity"
    out.mkdir(exist_ok=False)
    ranking.to_csv(out / "initial_ci_ranking.csv", index=False)
    shutil.copy2(base / "key_ratio_table9_reproduction.csv", out / "table9_reproduction_comparison.csv")
    rows = [record for record in base_raw.to_dict("records")]
    started = time.perf_counter()
    for ratio_index, ratio in enumerate(EXTENDED_RATIOS, 1):
        key_ids = key_ids_from_ranking(ranking, ratio, len(plans))
        ratio_out = out / f"ratio_{round(ratio * 100):03d}"
        ratio_out.mkdir()
        pd.DataFrame({"rank": range(1, len(key_ids) + 1), "flight_id": key_ids}).to_csv(ratio_out / "key_flights.csv", index=False)
        ratio_rows = []
        for optimizer_seed in range(3):
            row, _ = run_one(plans, conflicts, key_ids, cfg, grid, risk_map, ratio, optimizer_seed)
            rows.append(row); ratio_rows.append(row)
            print(f"[{(ratio_index - 1) * 3 + optimizer_seed + 1}/27] p={ratio:.2f} seed={optimizer_seed} final={row['remaining_conflicts']}", flush=True)
            pd.DataFrame(rows).to_csv(out / "raw_runs_extended.csv", index=False)
        pd.DataFrame(ratio_rows).to_csv(ratio_out / "raw_runs.csv", index=False)
    table = build_extended_table(pd.DataFrame(rows))
    table.to_csv(out / "extended_key_ratio_analysis.csv", index=False)
    _plot(table, "key_conflict_point_coverage", "Initial CI conflict point coverage", "Key ratio vs CI coverage", out / "key_ratio_vs_coverage.png")
    _plot(table, "stage1_remaining_conflicts_mean", "Stage1 remaining conflicts", "Key ratio vs Stage1 conflicts", out / "key_ratio_vs_stage1_conflicts.png", "stage1_remaining_conflicts_std")
    _plot(table, "final_remaining_conflicts_mean", "Final remaining conflicts", "Key ratio vs final conflicts", out / "key_ratio_vs_final_conflicts.png", "final_remaining_conflicts_std")
    _plot(table, "final_fitness_mean", "Final fitness", "Key ratio vs final fitness", out / "key_ratio_vs_fitness.png")
    _report(out / "extended_key_ratio_analysis_report.md", table)
    (out / "extended_manifest.json").write_text(json.dumps(dict(
        base_analysis=str(base), environment_seed=args.environment_seed, traffic_seed=args.traffic_seed,
        initial_deterministic_conflicts=len(deterministic), initial_uncertain_conflicts=len(conflicts),
        extended_ratios=EXTENDED_RATIOS, repeats=3, optimizer_seeds=[0, 1, 2], n_jobs=args.n_jobs,
        scope="reproduction-specific extended sensitivity analysis", total_runtime_sec=time.perf_counter() - started,
    ), indent=2), encoding="utf-8")
    print(f"output: {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
