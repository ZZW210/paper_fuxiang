"""One strict two-stage schedule using MultiGraph degree CI on frozen baseline plans."""
from __future__ import annotations

import copy
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.baseline_plans import BASELINE_SOURCE_COMMIT, baseline_metadata, baseline_plan_hash, load_baseline_plans
from src.config import load_config
from src.conflict_detection import count_conflict_pairs, detect_conflicts, write_conflicts_csv
from src.flight_plan import plot_routes_3d
from src.grid import AirspaceGrid
from src.paper_optimization import flight_strategies
from src.paper_scheduler import optimize_paper_schedule
from src.risk_map import generate_risk_map
from src.scene_diagnostics import coverage
from src.utils import set_random_seed
from src.visualization import write_fata_3d_html


EXPECTED_DETERMINISTIC = 99
EXPECTED_UNCERTAIN = 130
EXPECTED_PAIRS = 85
KEY_MODE = "multi_degree_ci"
CI_L = 2
KEY_COUNT = 10
OPTIMIZER_SEED = 0
DENSE_IDS = (52, 77, 70, 87)


def _pair(conflict) -> tuple[int, int]:
    return tuple(sorted((int(conflict.plan_a), int(conflict.plan_b))))


def _record_key(conflict) -> tuple[object, ...]:
    return (_pair(conflict), tuple(conflict.cell), int(conflict.idx_a), int(conflict.idx_b))


def _risk(plans, risk_map) -> float:
    return float(sum(float(risk_map[cell]) for plan in plans for cell in plan.path))


def _mean_speed(plan) -> float:
    return float(np.mean(plan.speed_profile or [0.0]))


def _is_changed(before, after) -> bool:
    return bool(abs(before.atd - after.atd) > 1e-6 or before.path != after.path or not np.allclose(before.speed_profile, after.speed_profile))


def build_multiline_network(plans, conflicts):
    """Return simple topology and true MultiGraph, one edge per conflict record."""
    topology = nx.Graph()
    multi = nx.MultiGraph()
    ids = [int(plan.id) for plan in plans]
    topology.add_nodes_from(ids)
    multi.add_nodes_from(ids)
    for index, conflict in enumerate(conflicts):
        a, b = int(conflict.plan_a), int(conflict.plan_b)
        topology.add_edge(a, b)
        multi.add_edge(a, b, conflict_index=index, cell=tuple(conflict.cell))
    return topology, multi


def multi_ci_l2(topology: nx.Graph, multi: nx.MultiGraph) -> dict[int, float]:
    multi_degree = dict(multi.degree())
    scores = {}
    for flight_id in topology.nodes:
        shell = [
            other for other, distance in nx.single_source_shortest_path_length(topology, flight_id, cutoff=CI_L).items()
            if distance == CI_L
        ]
        scores[flight_id] = float((multi_degree[flight_id] - 1) * sum(max(0, multi_degree[other] - 1) for other in shell))
    return scores


def multi_metrics(plans, conflicts):
    topology, multi = build_multiline_network(plans, conflicts)
    degree = dict(multi.degree())
    unique = dict(topology.degree())
    involvement = Counter(fid for conflict in conflicts for fid in (int(conflict.plan_a), int(conflict.plan_b)))
    if any(degree[plan.id] != involvement[plan.id] for plan in plans):
        raise AssertionError("MultiGraph degree must equal incident conflict-record count")
    ci = multi_ci_l2(topology, multi)
    ranked_ids = sorted(topology.nodes, key=lambda fid: (-ci[fid], fid))
    rank = {fid: index + 1 for index, fid in enumerate(ranked_ids)}
    rows = [
        dict(flight_id=plan.id, multi_degree=int(degree[plan.id]), unique_conflict_partners=int(unique[plan.id]),
             conflict_point_involvement=int(involvement[plan.id]), CI_multi_l2=ci[plan.id], multi_rank=rank[plan.id])
        for plan in plans
    ]
    return pd.DataFrame(rows).sort_values("multi_rank").reset_index(drop=True), topology, multi


def _plot_multiline_network(plans, multi, metrics: pd.DataFrame, output: Path) -> None:
    positions = {plan.id: ((plan.start[0] + plan.goal[0]) / 2, (plan.start[1] + plan.goal[1]) / 2) for plan in plans}
    values = metrics.set_index("flight_id")["multi_degree"].to_dict()
    fig, ax = plt.subplots(figsize=(10, 8))
    grouped = Counter(tuple(sorted((a, b))) for a, b, _ in multi.edges(keys=True))
    for (a, b), count in grouped.items():
        displayed = min(count, 12)
        for index in range(displayed):
            radius = (index - (displayed - 1) / 2) * 0.045
            nx.draw_networkx_edges(multi, positions, edgelist=[(a, b)], connectionstyle=f"arc3,rad={radius}", alpha=0.16, width=0.65, ax=ax)
    nodes = nx.draw_networkx_nodes(
        multi, positions, node_size=82, node_color=[values[node] for node in multi.nodes], cmap="YlOrRd",
        vmin=0, vmax=max(1, max(values.values())), ax=ax,
    )
    labels = {fid: str(fid) for fid in DENSE_IDS}
    nx.draw_networkx_labels(multi, positions, labels=labels, font_size=9, font_weight="bold", ax=ax)
    fig.colorbar(nodes, ax=ax, shrink=0.78, label="Multi-degree (all incident conflict records)")
    ax.set_title("Baseline conflict MultiGraph; visual parallel edges capped at 12 per flight pair")
    ax.set_xlabel("flight start/goal midpoint x (cell)")
    ax.set_ylabel("flight start/goal midpoint y (cell)")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_convergence(frame: pd.DataFrame, output: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(frame.stage_generation, frame.fitness, color="#3366AA", linewidth=1.8)
    axes[0].set(xlabel="Generation", ylabel="Best paper fitness", title=title)
    axes[1].plot(frame.stage_generation, frame.Nc, color="#C0392B", label="conflict points", linewidth=1.8)
    axes[1].plot(frame.stage_generation, frame.conflict_pairs, color="#16A085", label="conflict pairs", linewidth=1.5)
    axes[1].set(xlabel="Generation", ylabel="Conflicts", title="Best conflict trajectory")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _provenance(initial, final) -> pd.DataFrame:
    initial_records = {_record_key(conflict) for conflict in initial}
    initial_pairs = {_pair(conflict) for conflict in initial}
    rows = []
    for conflict in final:
        pair = _pair(conflict)
        exact = _record_key(conflict) in initial_records
        category = "exact_survivor" if exact else ("moved_same_pair" if pair in initial_pairs else "new_pair")
        rows.append(dict(
            flight_a=pair[0], flight_b=pair[1], cell_x=conflict.cell[0], cell_y=conflict.cell[1], cell_z=conflict.cell[2],
            idx_a=conflict.idx_a, idx_b=conflict.idx_b, time_a=conflict.time_a, time_b=conflict.time_b,
            time_diff=conflict.time_diff, required_sep=conflict.required_sep, provenance=category,
        ))
    return pd.DataFrame(rows)


def _dense_rows(initial, stage1, final, initial_conflicts, stage1_conflicts, final_conflicts, metrics, keys) -> pd.DataFrame:
    maps = [{plan.id: plan for plan in plans} for plans in (initial, stage1, final)]
    conflict_sets = [Counter(fid for conflict in conflicts for fid in (conflict.plan_a, conflict.plan_b)) for conflicts in (initial_conflicts, stage1_conflicts, final_conflicts)]
    rank_map = metrics.set_index("flight_id")
    rows = []
    for flight_id in DENSE_IDS:
        before, after1, after2 = (plans[flight_id] for plans in maps)
        row = rank_map.loc[flight_id]
        rows.append(dict(
            flight_id=flight_id, is_multi_top10=flight_id in keys, multi_rank=int(row.multi_rank), multi_degree=int(row.multi_degree),
            unique_conflict_partners=int(row.unique_conflict_partners), conflict_point_involvement=int(row.conflict_point_involvement), CI_multi_l2=float(row.CI_multi_l2),
            stage1_changed=_is_changed(before, after1), stage1_primary_strategy="+".join(str(v) for v in flight_strategies(after1)) or "none",
            atd_before=before.atd, atd_after_stage1=after1.atd, atd_after_stage2=after2.atd,
            speed_before=_mean_speed(before), speed_after_stage1=_mean_speed(after1), speed_after_stage2=_mean_speed(after2),
            reroute_stage1=bool(after1.rerouted or after1.path != before.path), reroute_final=bool(after2.rerouted or after2.path != before.path),
            conflicts_initial=int(conflict_sets[0][flight_id]), conflicts_after_stage1=int(conflict_sets[1][flight_id]), conflicts_after_stage2=int(conflict_sets[2][flight_id]),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parent
    baseline = baseline_metadata(root)
    if baseline_plan_hash(root) != baseline["sha256"]:
        raise AssertionError("Baseline hash validation failed before scheduling")
    cfg = load_config(root / "config.yaml", {"optimization": {"scheduler_mode": "paper_strict", "n_jobs": 8}})
    cfg["conflict"]["t_conflict"] = 30.0
    cfg["flight"]["random_seed"] = OPTIMIZER_SEED
    cfg["baseline_plans"] = baseline
    plans = load_baseline_plans(root)
    grid = AirspaceGrid.from_config(cfg, seed=cfg.get("environment_seed", 2025))
    risk_map = generate_risk_map(grid, cfg)
    deterministic = detect_conflicts(plans, cfg, uncertain=False)
    initial = detect_conflicts(plans, cfg, uncertain=True)
    if (len(deterministic), len(initial), count_conflict_pairs(initial)) != (EXPECTED_DETERMINISTIC, EXPECTED_UNCERTAIN, EXPECTED_PAIRS):
        raise RuntimeError(f"Baseline conflict validation failed: {len(deterministic)}/{len(initial)}/{count_conflict_pairs(initial)}")
    metrics, topology, multi = multi_metrics(plans, initial)
    keys = metrics.head(KEY_COUNT).flight_id.astype(int).tolist()
    if len(keys) != KEY_COUNT or keys != sorted(topology.nodes, key=lambda fid: (-metrics.set_index("flight_id").loc[fid, "CI_multi_l2"], fid))[:KEY_COUNT]:
        raise AssertionError("Stage1 keys are not the Multi-degree CI l=2 Top10")
    key_point_coverage, key_pair_coverage = coverage(initial, keys)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_id = f"{stamp}_baseline_multiline_seed0"
    out = root / "outputs" / "multiline_baseline_full_run" / run_id
    out.mkdir(parents=True, exist_ok=False)
    metrics.head(20).assign(rank=range(1, 21)).to_csv(out / "multi_top20.csv", index=False)
    metrics[metrics.flight_id.isin(keys)].sort_values("multi_rank").to_csv(out / "key_flights_multi.csv", index=False)
    print("key_flight_mode = multi_degree_ci", flush=True)
    print(metrics.head(20).to_string(index=False), flush=True)
    for flight_id in DENSE_IDS:
        value = metrics[metrics.flight_id.eq(flight_id)].iloc[0]
        print(f"{flight_id}: degree={value.multi_degree}, partners={value.unique_conflict_partners}, involvement={value.conflict_point_involvement}, CI={value.CI_multi_l2}, rank={value.multi_rank}", flush=True)
    _plot_multiline_network(plans, multi, metrics, out / "conflict_network_multiline.png")
    plot_routes_3d(grid, plans, out / "initial_routes.png", "Frozen baseline initial routes")
    write_fata_3d_html(grid, plans, initial, out / "fata_3d_before.html", "Frozen baseline before scheduling: Multi CI Top10", keys)
    start = time.perf_counter()
    set_random_seed(OPTIMIZER_SEED)
    result = optimize_paper_schedule(copy.deepcopy(plans), initial, keys, cfg, grid, risk_map, progress=True)
    final = detect_conflicts(result.final.plans, cfg, uncertain=True)
    if baseline_plan_hash(root) != baseline["sha256"]:
        raise AssertionError("Baseline hash changed during scheduling")
    stage1 = result.stage1.conflicts
    stage1_pairs, final_pairs = count_conflict_pairs(stage1), count_conflict_pairs(final)
    initial_risk = _risk(plans, risk_map)
    final_risk = _risk(result.final.plans, risk_map)
    provenance = _provenance(initial, final)
    provenance.to_csv(out / "final_conflict_provenance.csv", index=False)
    counts = provenance.provenance.value_counts().to_dict() if not provenance.empty else {}
    dense = _dense_rows(plans, result.stage1.plans, result.final.plans, initial, stage1, final, metrics, set(keys))
    dense.to_csv(out / "dense_flights_52_77_70_87.csv", index=False)
    convergence = pd.DataFrame(result.convergence)
    stage1_curve = convergence[convergence.stage.eq("stage1")].copy()
    stage2_curve = convergence[convergence.stage.eq("stage2")].copy()
    stage1_curve.to_csv(out / "stage1_convergence.csv", index=False)
    stage2_curve.to_csv(out / "stage2_convergence.csv", index=False)
    _plot_convergence(stage1_curve, out / "stage1_convergence.png", "Stage1 FATA convergence")
    if not stage2_curve.empty:
        _plot_convergence(stage2_curve, out / "stage2_convergence.png", "Stage2 ADM + FATA convergence")
    write_fata_3d_html(grid, result.stage1.plans, stage1, out / "fata_3d_after_stage1.html", "Frozen baseline after Stage1", keys)
    write_fata_3d_html(grid, result.final.plans, final, out / "fata_3d_after.html", "Frozen baseline after Stage2", keys)
    write_conflicts_csv(final, out / "final_conflicts.csv")
    stage1_metrics = dict(
        initial_conflict_points=len(initial), initial_conflict_pairs=count_conflict_pairs(initial), key_flights=keys,
        key_conflict_point_coverage=key_point_coverage, key_conflict_pair_coverage=key_pair_coverage,
        stage1_final_conflict_points=len(stage1), stage1_final_conflict_pairs=stage1_pairs,
        stage1_conflict_reduction=1 - len(stage1) / len(initial),
        stage1_changed_flights=sum(_is_changed(a, b) for a, b in zip(plans, result.stage1.plans)),
        stage1_delayed_flights=result.stage1.components["n_delay"],
        stage1_risk_increase_percent=(_risk(result.stage1.plans, risk_map) - initial_risk) / initial_risk * 100,
        stage1_fitness=result.stage1.fitness, stage1_runtime_sec=result.stage1_seconds, stage1_dimension=result.dimensions[0],
    )
    stage2_metrics = dict(
        stage2_input_conflict_points=len(stage1), stage2_input_conflict_pairs=stage1_pairs,
        final_conflict_points=len(final), final_conflict_pairs=final_pairs,
        stage2_reduction_ratio=1 - len(final) / max(1, len(stage1)), zero_conflict=not final,
        final_exact_survivors=int(counts.get("exact_survivor", 0)), final_moved_same_pair=int(counts.get("moved_same_pair", 0)),
        final_new_conflict_points=int(counts.get("new_pair", 0)), final_new_conflict_pairs=len({_pair(c) for c in final if _pair(c) not in {_pair(old) for old in initial}}),
        changed_flights=sum(_is_changed(a, b) for a, b in zip(plans, result.final.plans)), delayed_flights=result.final.components["n_delay"],
        risk_increase_percent=(final_risk - initial_risk) / initial_risk * 100, final_paper_fitness=result.final.fitness,
        stage2_runtime_sec=result.stage2_seconds, total_runtime_sec=time.perf_counter() - start, stage2_dimension=result.dimensions[1],
    )
    (out / "stage1_metrics.json").write_text(json.dumps(stage1_metrics, indent=2), encoding="utf-8")
    (out / "stage2_metrics.json").write_text(json.dumps(stage2_metrics, indent=2), encoding="utf-8")
    manifest = dict(
        run_id=run_id, baseline_source_commit=BASELINE_SOURCE_COMMIT, baseline_plan_hash=baseline["sha256"],
        conflict_parameters=cfg["conflict"], key_flight_mode=KEY_MODE, ci_l=CI_L, important_ratio=0.10,
        optimizer_seed=OPTIMIZER_SEED, NP=cfg["fata"]["NP"], Ngen=cfg["fata"]["Ngen_max_stage1"], Parf=cfg["fata"]["Parf"],
        initial_deterministic_conflicts=len(deterministic), initial_uncertain_conflicts=len(initial), initial_conflict_pairs=count_conflict_pairs(initial),
        multi_top10=keys, key_conflict_point_coverage=key_point_coverage, stage1_remaining=len(stage1), final_remaining=len(final),
        risk_increase_percent=stage2_metrics["risk_increase_percent"], delayed_flights=stage2_metrics["delayed_flights"], changed_flights=stage2_metrics["changed_flights"],
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    dense_summary = "\n".join(
        f"- Flight {row.flight_id}: multi degree={int(row.multi_degree)}, CI_multi_l2={row.CI_multi_l2:.1f}, "
        f"rank={int(row.multi_rank)}, Multi Top10={bool(row.is_multi_top10)}."
        for row in dense.itertuples(index=False)
    )
    dense_key_count = int(dense.is_multi_top10.sum())
    multiline_finding = (
        "Multi-degree correctly exposes the repeated-conflict incident count, but this fixed l=2 CI did not select "
        "the dense flights in this baseline because their simple-topology two-hop shell contribution is zero or small."
        if dense_key_count == 0 else
        "Multi-degree CI selected part of the dense-flight set in this baseline."
    )
    report = f"""# Multi-degree CI Full Baseline Run

1. Frozen baseline loaded: yes. Source commit `{BASELINE_SOURCE_COMMIT}`.
2. Baseline hash validated before and after the run: `{baseline['sha256']}`.
3. Initial deterministic/uncertain conflict points: {len(deterministic)} / {len(initial)}; unique pairs: {count_conflict_pairs(initial)}.
4. Multi CI l=2 Top10: `{keys}`.
5. Dense-flight results:
{dense_summary}
6. Multi Top10 covers {int(round(key_point_coverage * len(initial)))} / {len(initial)} conflict points ({key_point_coverage:.4%}); unique-pair coverage is {key_pair_coverage:.4%}.
7. Stage1: {len(initial)} -> {len(stage1)} ({stage1_metrics['stage1_conflict_reduction']:.4%} reduction).
8. Stage2: {len(stage1)} -> {len(final)} ({stage2_metrics['stage2_reduction_ratio']:.4%} reduction).
9. Zero conflict: {stage2_metrics['zero_conflict']}.
10. Final provenance: exact survivors={stage2_metrics['final_exact_survivors']}, moved same pair={stage2_metrics['final_moved_same_pair']}, new-pair points={stage2_metrics['final_new_conflict_points']}.
11. Risk increase: {stage2_metrics['risk_increase_percent']:.4f}%; delayed flights={stage2_metrics['delayed_flights']}; changed flights={stage2_metrics['changed_flights']}.
12. Runtime: Stage1={result.stage1_seconds:.3f}s; Stage2={result.stage2_seconds:.3f}s; total={stage2_metrics['total_runtime_sec']:.3f}s.
13. {multiline_finding} Multi-degree CI explicitly uses full incident-record degree; this run reports the actual scheduling outcome and does not alter the paper-strict optimizer to favor that conclusion.
"""
    (out / "multiline_full_run_report.md").write_text(report, encoding="utf-8")
    print(f"BASELINE HASH: {baseline['sha256']}")
    print(f"INITIAL: deterministic = {len(deterministic)}, uncertain = {len(initial)}, pairs = {count_conflict_pairs(initial)}")
    print(f"MULTI TOP10: {keys}")
    print(f"COVERAGE: covered = {int(round(key_point_coverage * len(initial)))} / {len(initial)}, coverage = {key_point_coverage:.2%}")
    print(f"SCHEDULE: {len(initial)} => Stage1 {len(stage1)} => Final {len(final)}")
    print(f"ZERO CONFLICT: {not final}")
    print(f"RISK INCREASE: {stage2_metrics['risk_increase_percent']:.4f} %")
    print(f"DELAYED FLIGHTS: {stage2_metrics['delayed_flights']}")
    print(f"CHANGED FLIGHTS: {stage2_metrics['changed_flights']}")
    print(f"RUNTIME: Stage1 {result.stage1_seconds:.3f} s, Stage2 {result.stage2_seconds:.3f} s, Total {stage2_metrics['total_runtime_sec']:.3f} s")
    print(out)


if __name__ == "__main__":
    main()
