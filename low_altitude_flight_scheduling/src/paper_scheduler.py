"""Two-stage paper scheduler and reproducible scientific diagnostics."""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import fata
from .adm_matching import ADMResult, adm_fata_optimize
from .conflict_detection import (detect_conflicts, count_conflict_pairs, write_calibration_report,
                                 write_conflict_diagnostics)
from .conflict_network import (build_conflict_network, network_metrics, run_attack_suite,
                               select_paper_key_flights, write_network_outputs)
from .flight_plan import generate_flight_plans, write_flight_generation_report
from .baseline_plans import load_baseline_plans
from .grid import AirspaceGrid
from .paper_optimization import (PaperPopulationObjective, PaperReference, PaperEvaluation,
                                build_stage1_decision_layout, build_stage2_decision_layout,
                                paper_objective_components, paper_fitness, conflict_weight_delta, flight_strategies)
from .paper_performance import PerformanceCounters
from .risk_map import generate_risk_map
from .utils import ensure_dir, set_random_seed
from .visualization import write_all_route_visuals, write_strategy_overview_html


@dataclass
class PaperScheduleResult:
    stage1: PaperEvaluation
    final: PaperEvaluation
    adm: ADMResult | None
    stage1_seconds: float
    stage2_seconds: float
    convergence: list[dict]
    performance: dict = field(default_factory=dict)
    dimensions: tuple = (0, 0)
    fitness_evaluations: tuple = (0, 0)


def optimize_paper_schedule(plans, conflicts, key_ids, cfg, grid, risk_map, progress=True):
    reference = PaperReference.from_initial(plans, risk_map, conflicts, cfg["optimization"]["t_delay_max"])
    generations1 = int(cfg["fata"]["Ngen_max_stage1"])
    generations2 = int(cfg["fata"]["Ngen_max_stage2"])
    layout1 = build_stage1_decision_layout(plans, key_ids, conflicts, cfg, grid)
    objective1 = PaperPopulationObjective(plans, plans, layout1, cfg, grid, risk_map, reference, generations1, 1)
    trace = []
    performance = PerformanceCounters()
    evaluations1, dimension2 = 0, 0

    def record(stage, objective, offset):
        def callback(position, score, generation, context):
            evaluation = objective.evaluation(position, generation, context)
            trace.append(dict(global_generation=offset + generation, stage=stage,
                              stage_generation=generation, fitness=evaluation.fitness,
                              **evaluation.components, delta=evaluation.delta))
            if progress and (generation == 1 or generation % 10 == 0 or generation == objective.max_gen):
                print(f"{stage} generation {generation}/{objective.max_gen}: "
                      f"fitness={score:.6f}, Nc={evaluation.components['Nc']}", flush=True)
            return {"remaining_conflicts": evaluation.components["Nc"], "delay_count": evaluation.components["n_delay"]}
        return callback

    start = time.perf_counter()
    if layout1.dim:
        result1 = fata.fata_optimize_paper(
            objective1.fitness, layout1.lower, layout1.upper, layout1.dim,
            population=cfg["fata"]["NP"], max_iter=generations1, seed=cfg["flight"]["random_seed"],
            parf=cfg["fata"]["Parf"], n_jobs=cfg["optimization"]["n_jobs"],
            callback=record("stage1", objective1, 0),
            vectorized_update=cfg.get("paper_performance", {}).get("fata_vectorized_update", False),
        )
        if not np.isfinite(result1.best_fitness):
            raise RuntimeError("Stage 1 found no geometrically feasible candidate")
        stage1 = objective1.evaluation(result1.best_position, generations1)
        performance.add(result1.performance or {})
        evaluations1 = result1.fitness_evaluations
    else:
        components = paper_objective_components(plans, plans, risk_map, conflicts, cfg)
        stage1 = PaperEvaluation(paper_fitness(components, reference, cfg, generations1, generations1),
                                components, plans, conflicts, 0.9)
    seconds1 = time.perf_counter() - start
    performance.add(objective1.profile.snapshot())
    if progress:
        print(f"Stage 1 remaining conflicts: {stage1.components['Nc']}", flush=True)
        print(f"Stage 2 remaining conflict points: {len(stage1.conflicts)}", flush=True)
    start = time.perf_counter()
    adm = None
    if stage1.conflicts:
        layout2 = build_stage2_decision_layout(stage1.plans, stage1.conflicts, cfg, grid, plans)
        dimension2 = layout2.dim
        objective2 = PaperPopulationObjective(stage1.plans, plans, layout2, cfg, grid, risk_map, reference, generations2, 2)
        adm = adm_fata_optimize(stage1.plans, plans, stage1.conflicts, layout2, cfg, grid, risk_map,
                                reference, seed=cfg["flight"]["random_seed"] + 206,
                                callback=record("stage2", objective2, generations1))
        with objective2.profile.measure("conflict_detection"):
            final_conflicts = detect_conflicts(adm.best_plans, cfg, uncertain=True)
        with objective2.profile.measure("objective"):
            components = paper_objective_components(adm.best_plans, plans, risk_map, final_conflicts, cfg)
        final = PaperEvaluation(paper_fitness(components, reference, cfg, generations2, generations2),
                                components, adm.best_plans, final_conflicts,
                                conflict_weight_delta(generations2, generations2, cfg["fata"]["gamma"]))
        performance.add(adm.performance or {})
        performance.add(objective2.profile.snapshot())
    else:
        if progress:
            print("Stage 2 skipped: no remaining conflicts.", flush=True)
        final = stage1
    return PaperScheduleResult(stage1, final, adm, seconds1, time.perf_counter() - start, trace,
                               performance.snapshot(), (layout1.dim, dimension2),
                               (evaluations1, adm.fitness_evaluations if adm else 0))


def _strategy_name(values):
    names = ("schedule", "speed", "reroute")
    if isinstance(values, (int, np.integer)):
        return names[int(values)]
    return '+'.join(names[int(s)] for s in values)


def _counts(assignments):
    return {name: sum(i in active for active in assignments.values()) for i, name in enumerate(("schedule", "speed", "reroute"))}


def timing_shift_diagnostics(initial, final):
    original = {p.id: p for p in initial}
    shifts = np.asarray([p.atd - original[p.id].etd for p in final])
    return dict(advanced_flight_count=int(np.sum(shifts < -1e-6)),
                delayed_flight_count=int(np.sum(shifts > 1e-6)),
                mean_absolute_atd_shift=float(np.mean(np.abs(shifts))) if len(shifts) else 0.0,
                max_absolute_atd_shift=float(np.max(np.abs(shifts))) if len(shifts) else 0.0)


def _physically_changed(before, after):
    return abs(before.atd - after.atd) > 1e-6 or before.path != after.path or len(before.speed_profile) != len(after.speed_profile) or not np.allclose(before.speed_profile, after.speed_profile, atol=1e-6, rtol=0.0)


def write_paper_diagnostics(out, result, initial, key_ids, ranked_metrics, cfg):
    original = {p.id: p for p in initial}
    ranks = {int(fid): i + 1 for i, fid in enumerate(ranked_metrics["flight_id"])}
    assignment = []
    for plan in result.stage1.plans:
        if plan.id not in key_ids:
            continue
        base = original[plan.id]
        changed_segments = sum(abs(a - b) > 1e-6 for a, b in zip(plan.speed_profile, base.speed_profile)) if not plan.rerouted else 0
        assignment.append(dict(flight_id=plan.id, CI_rank=ranks[plan.id], strategy=_strategy_name(flight_strategies(plan)),
                               ATD=plan.atd, changed_segments=changed_segments, rerouted=plan.rerouted,
                               fitness=result.stage1.fitness))
    pd.DataFrame(assignment, columns=["flight_id", "CI_rank", "strategy", "ATD", "changed_segments", "rerouted", "fitness"]).to_csv(out / "paper_stage1_strategy_assignment.csv", index=False)
    conflict_rows, probability_rows = [], []
    if result.adm is not None:
        adm = result.adm
        records = [(g, actors, sampled, adm.probability_history[g - 1], "generation_best")
                   for g, actors, sampled in (adm.generation_history or [])]
        records.append((adm.best_generation, adm.best_decision_vector[:len(result.stage1.conflicts)], adm.best_strategies,
                        adm.probability_history[max(0, adm.best_generation - 1)], "selected_best"))
        for generation, actors, sampled, probability, record_type in records:
            for i, conflict in enumerate(result.stage1.conflicts):
                p = probability[i]
                conflict_rows.append(dict(conflict_id=i, plan_a=conflict.plan_a, plan_b=conflict.plan_b,
                                          cell_x=conflict.cell[0], cell_y=conflict.cell[1], cell_z=conflict.cell[2],
                                          P_schedule=p[0], P_speed=p[1], P_reroute=p[2],
                                          sampled_strategy=_strategy_name(sampled[i]), actor_gene=actors[i],
                                          selected_actor=conflict.plan_a if actors[i] < 1 else conflict.plan_b,
                                          stage2_generation=generation, record_type=record_type))
        for generation, matrix in enumerate(adm.probability_history):
            for i, p in enumerate(matrix):
                probability_rows.append(dict(generation=generation, conflict_id=i, P_schedule=p[0], P_speed=p[1], P_reroute=p[2]))
    conflict_columns = ["conflict_id", "plan_a", "plan_b", "cell_x", "cell_y", "cell_z", "P_schedule", "P_speed", "P_reroute",
                        "sampled_strategy", "actor_gene", "selected_actor", "stage2_generation", "record_type"]
    pd.DataFrame(conflict_rows, columns=conflict_columns).to_csv(out / "paper_stage2_conflict_strategy.csv", index=False)
    stage1_map = {p.id: p for p in result.stage1.plans}
    history = []
    for plan in result.final.plans:
        stage1_active = flight_strategies(stage1_map[plan.id])
        stage2_active = result.adm.best_flight_strategies.get(plan.id, ()) if result.adm else ()
        counts = {name: sum(r["selected_actor"] == plan.id and r["sampled_strategy"] == name and r["record_type"] == "selected_best" for r in conflict_rows)
                  for name in ("schedule", "speed", "reroute")}
        row = {"flight_id": plan.id, "is_key_flight": int(plan.id in key_ids),
               "stage1_strategy": _strategy_name(stage1_active) if stage1_active else "none",
               **{f"stage2_{name}_conflict_count": value for name, value in counts.items()},
               "final_strategy_combination": _strategy_name(flight_strategies(plan)) or "none"}
        for stage, active in (("stage1", stage1_active), ("stage2", stage2_active), ("final", flight_strategies(plan))):
            row.update({f"{stage}_{name}": int(i in active) for i, name in enumerate(("schedule", "speed", "reroute"))})
        row.update({f"final_uses_{name}": int(i in flight_strategies(plan)) for i, name in enumerate(("schedule", "speed", "reroute"))})
        history.append(row)
    pd.DataFrame(history).to_csv(out / "paper_flight_strategy_history.csv", index=False)
    pd.DataFrame(probability_rows, columns=["generation", "conflict_id", "P_schedule", "P_speed", "P_reroute"]).to_csv(out / "adm_probability_history.csv", index=False)
    columns = ["global_generation", "stage", "stage_generation", "fitness", "Nc", "Tdelay", "Tair", "ORISK", "delta", "n_delay", "n_battery"]
    pd.DataFrame(result.convergence, columns=columns).to_csv(out / "paper_convergence.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4))
    for stage in ("stage1", "stage2"):
        rows = [r for r in result.convergence if r["stage"] == stage]
        if rows:
            ax.plot([r["global_generation"] for r in rows], [r["fitness"] for r in rows], label=stage)
    boundary = cfg["fata"]["Ngen_max_stage1"]
    ax.axvline(boundary, color="gray", linestyle="--", label="Stage 1 -> Stage 2")
    ax.set(xlabel="Generation", ylabel="Paper fitness")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "paper_convergence.png", dpi=160)
    fig.savefig(out / "fata_convergence.png", dpi=160)
    plt.close(fig)
    with (out / "paper_run_config.json").open("w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)


def run_paper_main(cfg, args, root: Path):
    start = time.perf_counter()
    output_profile = PerformanceCounters()
    out = ensure_dir(root / args.outputs)
    set_random_seed(args.seed)
    # Table 1: strict scheduling uses 30 s, not the legacy calibration's 20 s.
    cfg["conflict"]["t_conflict"] = 30.0
    grid = AirspaceGrid.from_config(cfg, seed=cfg.get("environment_seed", args.seed))
    risk_map = generate_risk_map(grid, cfg, out)
    baseline = cfg.get("baseline_plans")
    if baseline and int(cfg["flight"]["n_flights"]) == 100:
        plans = load_baseline_plans(root)
        print(f"Using immutable baseline plans: {baseline['sha256']}", flush=True)
    else:
        plans = generate_flight_plans(grid, risk_map, cfg, out, seed=cfg.get("traffic_seed", args.seed))
    for plan in plans:
        if plan.etd < 1.0:
            shift = 1.0 - plan.etd
            plan.etd = 1.0
            plan.eta_times = [t + shift for t in plan.eta_times]
        plan.scheduled_etd = plan.etd
    with (out / "initial_plans.pkl").open("wb") as fh:
        pickle.dump(plans, fh)
    conflicts_no = detect_conflicts(plans, cfg, uncertain=False, output_csv=out / "conflicts_no_uncertain.csv")
    conflicts = detect_conflicts(plans, cfg, uncertain=True, output_csv=out / "conflicts_uncertain.csv")
    write_flight_generation_report(out, plans, grid, len(conflicts_no), len(conflicts), conflicts=conflicts)
    write_calibration_report(len(conflicts_no), len(conflicts), out)
    graph = build_conflict_network(plans, conflicts)
    metrics = network_metrics(graph, ci_l=cfg["network"]["ci_l"])
    ranked = metrics.sort_values(["collective_influence", "flight_id"], ascending=[False, True], kind="stable")
    # paper_strict uses only this ratio; stage1_key_ratio is legacy-engineering configuration.
    key_ids = select_paper_key_flights(metrics, len(plans), cfg["optimization"]["important_ratio"])
    attacks = run_attack_suite(graph, metrics)
    write_network_outputs(graph, ranked, attacks, out)
    print(f"Initial conflicts: {len(conflicts)}", flush=True)
    print(f"Stage 1 key flights: {len(key_ids)} {key_ids}", flush=True)
    print(f"paper_strict: NP={cfg['fata']['NP']}, generations={cfg['fata']['Ngen_max_stage1']}+"
          f"{cfg['fata']['Ngen_max_stage2']}, n_jobs={cfg['optimization']['n_jobs']}", flush=True)
    result = optimize_paper_schedule(plans, conflicts, key_ids, cfg, grid, risk_map)
    stage1_assignments = {p.id: flight_strategies(p) for p in result.stage1.plans if p.id in key_ids}
    stage2_assignments = result.adm.best_flight_strategies if result.adm else {}
    counts1, counts2 = _counts(stage1_assignments), _counts(stage2_assignments)
    summary = dict(
        scheduler_mode="paper_strict", seed=args.seed, n_jobs=cfg["optimization"]["n_jobs"],
        baseline_plan_hash=baseline.get("sha256") if baseline else None,
        baseline_source_commit=baseline.get("source_commit") if baseline else None,
        paper_objective_scale_mode=cfg["optimization"]["paper_objective_scale_mode"],
        run_id=cfg.get("run", {}).get("run_id", "unarchived"),
        NP=cfg["fata"]["NP"], Ngen_max_stage1=cfg["fata"]["Ngen_max_stage1"], Ngen_max_stage2=cfg["fata"]["Ngen_max_stage2"],
        stage2_skipped=result.adm is None, initial_conflicts_uncertain=len(conflicts),
        initial_conflicts_without_uncertainty=len(conflicts_no), initial_conflict_points_uncertain=len(conflicts),
        initial_conflict_pairs_uncertain=count_conflict_pairs(conflicts),
        final_conflicts_two_stage=len(result.final.conflicts), final_conflicts_one_stage=len(result.stage1.conflicts),
        final_conflict_points_two_stage=len(result.final.conflicts), final_conflict_points_one_stage=len(result.stage1.conflicts),
        final_conflict_pairs_two_stage=count_conflict_pairs(result.final.conflicts),
        final_conflict_pairs_one_stage=count_conflict_pairs(result.stage1.conflicts),
        stage1_key_flight_count=len(key_ids), stage1_initial_conflict_points=len(conflicts),
        stage1_final_conflict_points=len(result.stage1.conflicts), stage2_initial_conflict_points=len(result.stage1.conflicts),
        stage2_final_conflict_points=len(result.final.conflicts), stage1_fata_time=result.stage1_seconds,
        stage2_fata_time=result.stage2_seconds, final_fitness=result.final.fitness, final_paper_fitness=result.final.fitness,
        changed_flight_count_two_stage=sum(p.changed for p in result.final.plans),
        changed_flight_count_one_stage=sum(p.changed for p in result.stage1.plans), delayed_flight_count=result.final.components["n_delay"],
        stage1_strategy_combination_count=sum(len(s) > 1 for s in stage1_assignments.values()),
        stage2_strategy_combination_count=sum(len(s) > 1 for s in stage2_assignments.values()),
        stage1_schedule_only_count=sum(s == (0,) for s in stage1_assignments.values()),
        stage1_actual_rerouted_flight_count=sum(p.path != original.path for p, original in zip(result.stage1.plans, plans)),
        stage2_actual_rerouted_flight_count=sum(p.path != base.path for p, base in zip(result.final.plans, result.stage1.plans)),
        stage2_modified_flight_count=sum(_physically_changed(base, p) for base, p in zip(result.stage1.plans, result.final.plans)),
        stage2_newly_changed_flight_count=sum(not base.changed and p.changed for base, p in zip(result.stage1.plans, result.final.plans)),
        stage1_dimension=result.dimensions[0], stage2_dimension=result.dimensions[1],
        stage1_fitness_evaluations=result.fitness_evaluations[0], stage2_fitness_evaluations=result.fitness_evaluations[1],
        astar_calls=result.performance.get("astar_calls", 0), astar_cache_hits=result.performance.get("astar_cache_hits", 0),
        astar_cache_misses=result.performance.get("astar_cache_misses", 0), astar_seconds=result.performance.get("astar_seconds", 0),
        conflict_detection_calls=result.performance.get("conflict_detection_calls", 0),
        runtime_stage1=result.stage1_seconds, runtime_stage2=result.stage2_seconds,
    )
    summary.update(timing_shift_diagnostics(plans, result.final.plans))
    accesses = summary["astar_cache_hits"] + summary["astar_cache_misses"]
    summary["astar_cache_hit_rate"] = summary["astar_cache_hits"] / accesses if accesses else 0.0
    summary["large_advance_without_delay_observed"] = bool(result.final.components["n_delay"] == 0 and result.final.components["Tdelay"] > 20000)
    for key, value in result.final.components.items():
        if key != "Nc":
            summary[f"final_{key}"] = value
    for stage, counts in ((1, counts1), (2, counts2)):
        for name, count in counts.items():
            summary[f"stage{stage}_strategy_{name}_count"] = count
    with output_profile.measure("file_output"):
        write_paper_diagnostics(out, result, plans, key_ids, ranked, cfg)
    for name, saved in (("final_plans_one_stage.pkl", result.stage1.plans), ("final_plans_two_stage.pkl", result.final.plans)):
        with (out / name).open("wb") as fh:
            pickle.dump(list(saved), fh)
    write_conflict_diagnostics(result.final.conflicts, out)
    pd.DataFrame([dict(method="paper_two_stage_adm_fata", final_conflicts=len(result.final.conflicts), fitness=result.final.fitness),
                  dict(method="paper_stage1_three_strategy_fata", final_conflicts=len(result.stage1.conflicts), fitness=result.stage1.fitness)]).to_csv(out / "table_two_stage_vs_one_stage.csv", index=False)
    with output_profile.measure("visualization"):
        write_all_route_visuals(out, grid, plans, result.final.plans, result.stage1.plans, conflicts, conflicts_no, result.final.conflicts, key_ids)
    summary["runtime_seconds"] = time.perf_counter() - start
    summary["total_runtime"] = summary["runtime_seconds"]
    with output_profile.measure("visualization"):
        write_strategy_overview_html(out / "strategy_overview.html", summary, ranked, attacks,
                                      [r["fitness"] for r in result.convergence], grid=grid, initial=plans,
                                      optimized=result.final.plans, conflicts=result.final.conflicts, key_ids=key_ids)
    summary["runtime_seconds"] = time.perf_counter() - start
    summary["total_runtime"] = summary["runtime_seconds"]
    pd.DataFrame([summary]).to_csv(out / "metrics_summary.csv", index=False)
    for key, value in output_profile.snapshot().items():
        result.performance[key] = result.performance.get(key, 0) + value
    rows = []
    for component in ("fata_population_evaluation", "fata_population_update", "conflict_detection", "objective", "route_decode", "astar", "multiprocessing_serialization", "visualization", "file_output"):
        calls, seconds = result.performance.get(component + "_calls", 0), result.performance.get(component + "_seconds", 0)
        rows.append(dict(component=component, calls=calls, total_seconds=seconds, mean_ms=1000 * seconds / calls if calls else 0,
                         percentage=100 * seconds / summary["total_runtime"]))
    pd.DataFrame(rows).to_csv(out / "performance_profile.csv", index=False)
    from .paper_consistency import write_implementation_report, write_paper_html_report

    write_implementation_report(out, summary)
    write_paper_html_report(out, summary)
    print("Main metrics summary")
    print(f"Stage 1 strategies: {counts1}")
    print(f"Stage 1 combinations: {summary['stage1_strategy_combination_count']}")
    if result.adm:
        print("Stage 2 final ADM probabilities:")
        print(np.array2string(result.adm.final_probability, threshold=result.adm.final_probability.size, precision=4))
    print(f"Stage 2 strategies: {counts2}")
    print(f"Stage 2 combinations: {summary['stage2_strategy_combination_count']}")
    print(f"Final conflicts: {len(result.final.conflicts)}")
    for key, value in result.final.components.items():
        if key != "Nc":
            print(f"{key}: {value}")
    print(f"paper fitness: {result.final.fitness:.8f}")
    for key in ("paper_objective_scale_mode", "changed_flight_count_two_stage", "advanced_flight_count", "delayed_flight_count", "mean_absolute_atd_shift", "max_absolute_atd_shift", "large_advance_without_delay_observed"):
        print(f"{key}: {summary[key]}")
    print(f"Stage1 runtime: {result.stage1_seconds:.3f}")
    print(f"Stage2 runtime: {result.stage2_seconds:.3f}")
    print(f"Total runtime: {summary['total_runtime']:.3f}")
    print(f"outputs: {out}")
