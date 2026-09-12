"""Two-stage paper scheduler and reproducible scientific diagnostics."""
from __future__ import annotations

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import fata
from .adm_matching import ADMResult, Strategy, adm_fata_optimize
from .conflict_detection import (detect_conflicts, count_conflict_pairs, write_calibration_report,
                                 write_conflict_diagnostics)
from .conflict_network import (build_conflict_network, network_metrics, run_attack_suite,
                               select_paper_key_flights, write_network_outputs)
from .flight_plan import generate_flight_plans, write_flight_generation_report
from .grid import AirspaceGrid
from .paper_optimization import (PaperPopulationObjective, PaperReference, PaperEvaluation,
                                build_stage1_decision_layout, build_stage2_decision_layout,
                                paper_objective_components, paper_fitness, conflict_weight_delta)
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


def optimize_paper_schedule(plans, conflicts, key_ids, cfg, grid, risk_map, progress=True):
    reference = PaperReference.from_initial(plans, risk_map, conflicts, cfg["optimization"]["t_delay_max"])
    generations1 = int(cfg["fata"]["Ngen_max_stage1"])
    generations2 = int(cfg["fata"]["Ngen_max_stage2"])
    layout1 = build_stage1_decision_layout(plans, key_ids, conflicts, cfg, grid)
    objective1 = PaperPopulationObjective(plans, plans, layout1, cfg, grid, risk_map, reference, generations1, 1)
    trace = []

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
        )
        if not np.isfinite(result1.best_fitness):
            raise RuntimeError("Stage 1 found no geometrically feasible candidate")
        stage1 = objective1.evaluation(result1.best_position, generations1)
    else:
        components = paper_objective_components(plans, plans, risk_map, conflicts, cfg)
        stage1 = PaperEvaluation(paper_fitness(components, reference, cfg, generations1, generations1),
                                components, plans, conflicts, 0.9)
    seconds1 = time.perf_counter() - start
    if progress:
        print(f"Stage 1 remaining conflicts: {stage1.components['Nc']}", flush=True)
        print(f"Stage 2 remaining conflict points: {len(stage1.conflicts)}", flush=True)
    start = time.perf_counter()
    adm = None
    if stage1.conflicts:
        layout2 = build_stage2_decision_layout(plans, stage1.conflicts, cfg, grid)
        objective2 = PaperPopulationObjective(stage1.plans, plans, layout2, cfg, grid, risk_map, reference, generations2, 2)
        adm = adm_fata_optimize(stage1.plans, plans, stage1.conflicts, layout2, cfg, grid, risk_map,
                                reference, seed=cfg["flight"]["random_seed"] + 206,
                                callback=record("stage2", objective2, generations1))
        final_conflicts = detect_conflicts(adm.best_plans, cfg, uncertain=True)
        components = paper_objective_components(adm.best_plans, plans, risk_map, final_conflicts, cfg)
        final = PaperEvaluation(paper_fitness(components, reference, cfg, generations2, generations2),
                                components, adm.best_plans, final_conflicts,
                                conflict_weight_delta(generations2, generations2, cfg["fata"]["gamma"]))
    else:
        if progress:
            print("Stage 2 skipped: no remaining conflicts.", flush=True)
        final = stage1
    return PaperScheduleResult(stage1, final, adm, seconds1, time.perf_counter() - start, trace)


def _strategy_name(value):
    return Strategy(int(value)).name.lower()


def _counts(assignments):
    return {name: sum(int(s) == i for s in assignments.values()) for i, name in enumerate(("schedule", "speed", "reroute"))}


def write_paper_diagnostics(out, result, initial, key_ids, ranked_metrics, cfg):
    original = {p.id: p for p in initial}
    ranks = {int(fid): i + 1 for i, fid in enumerate(ranked_metrics["flight_id"])}
    assignment = []
    for plan in result.stage1.plans:
        if plan.id not in key_ids:
            continue
        base = original[plan.id]
        changed_segments = sum(abs(a - b) > 1e-6 for a, b in zip(plan.speed_profile, base.speed_profile)) if not plan.rerouted else 0
        assignment.append(dict(flight_id=plan.id, CI_rank=ranks[plan.id], strategy=_strategy_name(plan.paper_strategy),
                               ATD=plan.atd, changed_segments=changed_segments, rerouted=plan.rerouted,
                               fitness=result.stage1.fitness))
    pd.DataFrame(assignment, columns=["flight_id", "CI_rank", "strategy", "ATD", "changed_segments", "rerouted", "fitness"]).to_csv(out / "paper_stage1_strategy_assignment.csv", index=False)
    conflict_rows, probability_rows = [], []
    if result.adm is not None:
        adm = result.adm
        for i, conflict in enumerate(result.stage1.conflicts):
            p = adm.final_probability[i]
            conflict_rows.append(dict(conflict_id=i, plan_a=conflict.plan_a, plan_b=conflict.plan_b,
                                      cell_x=conflict.cell[0], cell_y=conflict.cell[1], cell_z=conflict.cell[2],
                                      P_schedule=p[0], P_speed=p[1], P_reroute=p[2],
                                      selected_strategy=_strategy_name(adm.best_strategies[i]), target_flight=adm.conflict_owners[i]))
        for generation, matrix in enumerate(adm.probability_history):
            for i, p in enumerate(matrix):
                probability_rows.append(dict(generation=generation, conflict_id=i, P_schedule=p[0], P_speed=p[1], P_reroute=p[2]))
    pd.DataFrame(conflict_rows, columns=["conflict_id", "plan_a", "plan_b", "cell_x", "cell_y", "cell_z", "P_schedule", "P_speed", "P_reroute", "selected_strategy", "target_flight"]).to_csv(out / "paper_stage2_conflict_strategy.csv", index=False)
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
    out = ensure_dir(root / args.outputs)
    set_random_seed(args.seed)
    # Table 1: strict scheduling uses 30 s, not the legacy calibration's 20 s.
    cfg["conflict"]["t_conflict"] = 30.0
    grid = AirspaceGrid.from_config(cfg, seed=args.seed)
    risk_map = generate_risk_map(grid, cfg, out)
    plans = generate_flight_plans(grid, risk_map, cfg, out, seed=args.seed)
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
    key_ids = select_paper_key_flights(metrics, len(plans))
    attacks = run_attack_suite(graph, metrics)
    write_network_outputs(graph, ranked, attacks, out)
    print(f"Initial conflicts: {len(conflicts)}", flush=True)
    print(f"Stage 1 key flights: {len(key_ids)} {key_ids}", flush=True)
    print(f"paper_strict: NP={cfg['fata']['NP']}, generations={cfg['fata']['Ngen_max_stage1']}+"
          f"{cfg['fata']['Ngen_max_stage2']}, n_jobs={cfg['optimization']['n_jobs']}", flush=True)
    result = optimize_paper_schedule(plans, conflicts, key_ids, cfg, grid, risk_map)
    stage1_assignments = {p.id: p.paper_strategy for p in result.stage1.plans if p.id in key_ids}
    stage2_assignments = result.adm.best_flight_strategies if result.adm else {}
    counts1, counts2 = _counts(stage1_assignments), _counts(stage2_assignments)
    summary = dict(
        scheduler_mode="paper_strict", seed=args.seed, n_jobs=cfg["optimization"]["n_jobs"],
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
    )
    for key, value in result.final.components.items():
        if key != "Nc":
            summary[f"final_{key}"] = value
    for stage, counts in ((1, counts1), (2, counts2)):
        for name, count in counts.items():
            summary[f"stage{stage}_strategy_{name}_count"] = count
    write_paper_diagnostics(out, result, plans, key_ids, ranked, cfg)
    for name, saved in (("final_plans_one_stage.pkl", result.stage1.plans), ("final_plans_two_stage.pkl", result.final.plans)):
        with (out / name).open("wb") as fh:
            pickle.dump(saved, fh)
    write_conflict_diagnostics(result.final.conflicts, out)
    pd.DataFrame([dict(method="paper_two_stage_adm_fata", final_conflicts=len(result.final.conflicts), fitness=result.final.fitness),
                  dict(method="paper_stage1_three_strategy_fata", final_conflicts=len(result.stage1.conflicts), fitness=result.stage1.fitness)]).to_csv(out / "table_two_stage_vs_one_stage.csv", index=False)
    write_all_route_visuals(out, grid, plans, result.final.plans, result.stage1.plans, conflicts, conflicts_no, result.final.conflicts, key_ids)
    summary["runtime_seconds"] = time.perf_counter() - start
    summary["total_runtime"] = summary["runtime_seconds"]
    write_strategy_overview_html(out / "strategy_overview.html", summary, ranked, attacks,
                                  [r["fitness"] for r in result.convergence], grid=grid, initial=plans,
                                  optimized=result.final.plans, conflicts=result.final.conflicts, key_ids=key_ids)
    summary["runtime_seconds"] = time.perf_counter() - start
    summary["total_runtime"] = summary["runtime_seconds"]
    pd.DataFrame([summary]).to_csv(out / "metrics_summary.csv", index=False)
    from .paper_consistency import write_implementation_report, write_paper_html_report

    write_implementation_report(out, summary)
    write_paper_html_report(out, summary)
    print("Main metrics summary")
    print(f"Stage 1 strategies: {counts1}")
    if result.adm:
        print("Stage 2 final ADM probabilities:")
        print(np.array2string(result.adm.final_probability, threshold=result.adm.final_probability.size, precision=4))
    print(f"Stage 2 strategies: {counts2}")
    print(f"Final conflicts: {len(result.final.conflicts)}")
    for key, value in result.final.components.items():
        if key != "Nc":
            print(f"{key}: {value}")
    print(f"paper fitness: {result.final.fitness:.8f}")
    print(f"Stage1 runtime: {result.stage1_seconds:.3f}")
    print(f"Stage2 runtime: {result.stage2_seconds:.3f}")
    print(f"Total runtime: {summary['total_runtime']:.3f}")
    print(f"outputs: {out}")
