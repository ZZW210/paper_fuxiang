"""Fixed-scene, fixed-seed Stage2 local-action rollback diagnostic."""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import copy
import argparse
import hashlib
import io
import json
import pickle
import subprocess
import time
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.adm_matching import adm_fata_optimize
from src.config import load_config
from src.conflict_detection import _conflict_row, count_conflict_pairs, detect_conflicts, write_conflicts_csv
from src.fata import fata_optimize_paper
from src.grid import AirspaceGrid
from src.paper_optimization import (PaperPopulationObjective, PaperReference,
                                    _valid_route, build_stage1_decision_layout, build_stage2_decision_layout)
from src.paper_scheduler import _physically_changed
from src.risk_map import generate_risk_map
from src.stage2_rollback import COLUMNS
from src.utils import set_random_seed
from src.visualization import write_fata_3d_html

SOURCE = "ebc5b9cfbf689eda0ee0b1baf7439def510d2cdf"
BASELINE_RUN = "20260914_130804_634_baseline_multiline_l1_seed0"
HASH = "5580bd2ad2cbdcea7dac0b3f8ac8dd990babce979aee4b274285986bc68facf7"
TOP10 = [52, 77, 87, 70, 41, 62, 16, 83, 4, 18]
ROOT = Path(__file__).resolve().parent


def git(*args):
    return subprocess.check_output(["git", "-c", f"safe.directory={ROOT.parent.as_posix()}", *args], cwd=ROOT.parent)


def source_file(relative):
    return git("show", f"{SOURCE}:low_altitude_flight_scheduling/{relative}")


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def pair(c):
    return tuple(sorted((int(c.plan_a), int(c.plan_b))))


def event(c):
    return pair(c), tuple(c.cell), int(c.idx_a), int(c.idx_b)


def provenance(initial, final):
    old_events, old_pairs = {event(c) for c in initial}, {pair(c) for c in initial}
    columns = ["flight_a", "flight_b", "cell_x", "cell_y", "cell_z", "idx_a", "idx_b", "provenance"]
    return pd.DataFrame([
        [*pair(c), *c.cell, c.idx_a, c.idx_b,
         "exact_survivor" if event(c) in old_events else "moved_same_pair" if pair(c) in old_pairs else "new_pair"]
        for c in final], columns=columns)


def merge_logs(directory):
    frames = [pd.read_csv(path) for path in sorted(directory.glob("actions_*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLUMNS)


def counts(actions):
    accepted = actions.accepted.astype(str).str.lower().eq("true")
    return dict(rollback_count=int((~accepted).sum()), accepted_action_count=int(accepted.sum()),
                pairs_increased_count=int(actions.rollback_reason.eq("pairs_increased").sum()),
                same_pairs_points_not_decreased_count=int(actions.rollback_reason.eq("same_pairs_points_not_decreased").sum()))


def audit_actions(actions):
    expected = ((actions.pairs_after < actions.pairs_before) |
                ((actions.pairs_after == actions.pairs_before) & (actions.points_after < actions.points_before)))
    assert (actions.accepted == expected).all(), "Logged action violates acceptance criterion"
    for _, group in actions.groupby("evaluation_id", sort=False):
        current = (60, 69)
        for row in group.itertuples():
            assert (row.pairs_before, row.points_before) == current, "Action chain is not restored correctly"
            if row.accepted:
                current = (row.pairs_after, row.points_after)
    return dict(action_count=len(actions), acceptance_rule_verified=True,
                every_candidate_starts_from_stage1=True, rollback_chain_continuity_verified=True)


def main():
    if git("branch", "--show-current").decode().strip() != "scheduling":
        raise RuntimeError("This diagnostic must run on scheduling")
    out = ROOT / "outputs" / "stage2_rollback_experiment" / (datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3] + "_rollback_seed0")
    out.mkdir(parents=True, exist_ok=False)
    baseline_dir = f"outputs/multiline_baseline_full_run/{BASELINE_RUN}"
    old1 = json.loads(source_file(f"{baseline_dir}/stage1_metrics.json"))
    old2 = json.loads(source_file(f"{baseline_dir}/stage2_metrics.json"))
    old_manifest = json.loads(source_file(f"{baseline_dir}/manifest.json"))
    core_files = ("adm_matching.py", "fata.py", "conflict_detection.py", "grid.py", "risk_map.py", "config.py", "astar_3d.py")
    for name in core_files:
        current = (ROOT / "src" / name).read_text(encoding="utf-8").replace("\r\n", "\n")
        frozen = source_file(f"src/{name}").decode("utf-8").replace("\r\n", "\n")
        if current != frozen:
            raise RuntimeError(f"Unrelated core implementation differs from frozen experiment: {name}")
    for name, value in (("no_rollback_stage1_metrics.json", old1), ("no_rollback_stage2_metrics.json", old2),
                        ("no_rollback_manifest.json", old_manifest)):
        save_json(out / name, value)
    baseline_bytes = source_file("data/baseline_initial_plans.pkl")
    if hashlib.sha256(baseline_bytes).hexdigest() != HASH:
        raise RuntimeError("Frozen baseline hash mismatch")
    (out / "baseline_initial_plans.pkl").write_bytes(baseline_bytes)
    (out / "baseline_config.yaml").write_bytes(source_file("config.yaml"))
    cfg = load_config(out / "baseline_config.yaml", {"optimization": {"scheduler_mode": "paper_strict", "n_jobs": 8}})
    cfg["conflict"]["t_conflict"] = 30.0
    cfg["flight"]["random_seed"] = 0
    plans = pickle.loads(baseline_bytes)
    assert len(plans) == 100 and [p.id for p in plans] == list(range(100))
    assert (cfg["fata"]["NP"], cfg["fata"]["Ngen_max_stage1"], cfg["fata"]["Ngen_max_stage2"], cfg["fata"]["Parf"]) == (50, 200, 200, 0.2)
    assert cfg["conflict"] == old_manifest["conflict_parameters"]
    grid = AirspaceGrid.from_config(cfg, seed=cfg.get("environment_seed", 2025))
    risk = generate_risk_map(grid, cfg)
    initial = detect_conflicts(plans, cfg, uncertain=True)
    assert (len(detect_conflicts(plans, cfg, uncertain=False)), len(initial), count_conflict_pairs(initial)) == (99, 130, 85)
    graph, multi = nx.Graph(), nx.MultiGraph()
    graph.add_nodes_from(p.id for p in plans)
    multi.add_nodes_from(graph.nodes)
    for c in initial:
        graph.add_edge(c.plan_a, c.plan_b)
        multi.add_edge(c.plan_a, c.plan_b)
    degree = dict(multi.degree())
    ci = {i: (degree[i] - 1) * sum(max(0, degree[j] - 1) for j in graph.neighbors(i)) for i in graph}
    keys = sorted(graph, key=lambda i: (-ci[i], i))[:10]
    assert keys == TOP10
    manifest = dict(run_id=out.name, branch="scheduling", source_snapshot_commit=SOURCE,
                    baseline_run_id=BASELINE_RUN, baseline_plan_hash=HASH,
                    baseline_source_commit=old_manifest["baseline_source_commit"],
                    diagnostic_only=True, paper_contains_rollback_claim=False,
                    initial_conflict_points=130, initial_conflict_pairs=85, deterministic_conflicts=99,
                    multi_top10=keys, ci_l=1, optimizer_seed=0, stage1_rng_seed=0, stage2_rng_seed=206,
                    acceptance="strict lexicographic (global pairs, global points); no fitness tie-break",
                    action_granularity="flight x strategy, existing union of incident conflict requirements; shared ATD/speed/reroute windows not split",
                    configuration=copy.deepcopy(cfg), status="stage1_verification")
    save_json(out / "manifest.json", manifest)
    print(f"OUTPUT: {out}\nINITIAL: 130 points / 85 pairs\nTOP10: {keys}", flush=True)
    reference = PaperReference.from_initial(plans, risk, initial, cfg["optimization"]["t_delay_max"])
    layout1 = build_stage1_decision_layout(plans, keys, initial, cfg, grid)
    obj1 = PaperPopulationObjective(plans, plans, layout1, cfg, grid, risk, reference, 200, 1)
    start = time.perf_counter()
    set_random_seed(0)
    res1 = fata_optimize_paper(obj1.fitness, layout1.lower, layout1.upper, layout1.dim,
                              population=50, max_iter=200, seed=0, parf=0.2, n_jobs=8,
                              vectorized_update=cfg.get("paper_performance", {}).get("fata_vectorized_update", False))
    stage1 = obj1.evaluation(res1.best_position, 200)
    seconds1 = time.perf_counter() - start
    m1 = dict(initial_conflict_points=130, initial_conflict_pairs=85, key_flights=keys,
              stage1_final_conflict_points=len(stage1.conflicts), stage1_final_conflict_pairs=count_conflict_pairs(stage1.conflicts),
              stage1_fitness=stage1.fitness, stage1_runtime_sec=seconds1, stage1_dimension=layout1.dim,
              rollback_enabled=False, stage1_matches_reference=False)
    save_json(out / "stage1_metrics.json", m1)
    if len(stage1.conflicts) != 69 or count_conflict_pairs(stage1.conflicts) != 60 or stage1.fitness != old1["stage1_fitness"]:
        manifest["status"] = "stopped_stage1_mismatch"
        save_json(out / "manifest.json", manifest)
        raise RuntimeError(f"Stage1 is not identical to reference: {m1}; Stage2 NOT run")
    m1["stage1_matches_reference"] = True
    save_json(out / "stage1_metrics.json", m1)
    (out / "stage1_plans.pkl").write_bytes(pickle.dumps(list(stage1.plans)))
    print(f"Stage1 VERIFIED: 130 -> 69, fitness={stage1.fitness}, runtime={seconds1:.3f}s", flush=True)
    cfg["optimization"]["stage2_global_rollback"] = True
    search_dir, final_dir = out / "search_actions", out / "selected_actions"
    search_dir.mkdir()
    final_dir.mkdir()
    cfg["optimization"]["stage2_rollback_log_dir"] = str(search_dir)
    layout2 = build_stage2_decision_layout(stage1.plans, stage1.conflicts, cfg, grid, plans)
    def progress(position, score, generation, context):
        if generation == 1 or generation % 10 == 0:
            print(f"Stage2 generation {generation}/200 fitness={score:.6f}", flush=True)
        return {}
    start = time.perf_counter()
    adm = adm_fata_optimize(stage1.plans, plans, stage1.conflicts, layout2, cfg, grid, risk,
                            reference, seed=206, callback=progress)
    seconds2 = time.perf_counter() - start
    cfg["optimization"]["stage2_rollback_log_dir"] = str(final_dir)
    obj2 = PaperPopulationObjective(stage1.plans, plans, layout2, cfg, grid, risk, reference, 200, 2)
    selected = obj2.evaluation(adm.best_decision_vector, 200, adm.best_strategies)
    assert selected.fitness == adm.best_fitness
    final = detect_conflicts(adm.best_plans, cfg, uncertain=True)
    assert (count_conflict_pairs(final), len(final)) <= (60, 69)
    actions, selected_actions = merge_logs(search_dir), merge_logs(final_dir)
    save_json(out / "action_audit.json", dict(search=audit_actions(actions), selected=audit_actions(selected_actions)))
    actions.to_csv(out / "stage2_action_log.csv", index=False)
    actions[actions.accepted.astype(str).str.lower().ne("true")].to_csv(out / "stage2_rollback_log.csv", index=False)
    selected_actions.to_csv(out / "stage2_selected_action_log.csv", index=False)
    prov = provenance(initial, final)
    prov.to_csv(out / "final_conflict_provenance.csv", index=False)
    categories = prov.provenance.value_counts().to_dict()
    risk0 = sum(float(risk[c]) for p in plans for c in p.path)
    risk_final = sum(float(risk[c]) for p in adm.best_plans for c in p.path)
    m2 = dict(final_conflict_points=len(final), final_conflict_pairs=count_conflict_pairs(final),
              final_exact_survivors=int(categories.get("exact_survivor", 0)),
              final_moved_same_pair=int(categories.get("moved_same_pair", 0)),
              final_new_conflict_points=int(categories.get("new_pair", 0)),
              final_new_conflict_pairs=len({pair(c) for c in final} - {pair(c) for c in initial}),
              risk_increase_percent=(risk_final / risk0 - 1) * 100,
              delayed_flights=int(selected.components["n_delay"]),
              changed_flights=sum(_physically_changed(a, b) for a, b in zip(plans, adm.best_plans)),
              final_paper_fitness=selected.fitness, stage2_runtime_sec=seconds2,
              stage2_runtime_increase_sec=seconds2-old2["stage2_runtime_sec"],
              stage2_runtime_ratio=seconds2/old2["stage2_runtime_sec"], zero_conflict=not final,
              stage2_dimension=layout2.dim, fitness_evaluations=adm.fitness_evaluations,
              adm_probability_updates=len(adm.probability_history)-1,
              count_scope="all search decodes including incumbent reevaluations and final ADM decode; not unique committed schedule actions",
              **counts(actions), selected_plan_action_counts=counts(selected_actions))
    old_final_bytes = source_file(f"{baseline_dir}/final_conflicts.csv")
    (out / "no_rollback_final_conflicts.csv").write_bytes(old_final_bytes)
    old_prov = pd.read_csv(io.BytesIO(source_file(f"{baseline_dir}/final_conflict_provenance.csv")))
    old_new_pairs = {tuple(sorted((int(row.flight_a), int(row.flight_b)))) for row in old_prov[old_prov.provenance.eq("new_pair")].itertuples()}
    m2["original_new_pairs_still_present"] = sorted(old_new_pairs & {pair(c) for c in final})
    save_json(out / "stage2_metrics.json", m2)
    write_conflicts_csv(final, out / "final_conflicts.csv")
    write_fata_3d_html(grid, adm.best_plans, final, out / "fata_3d_after.html", "Stage2 global rollback diagnostic", keys)
    (out / "final_plans.pkl").write_bytes(pickle.dumps(list(adm.best_plans)))
    pd.DataFrame(adm.final_probability, columns=["schedule", "speed", "reroute"]).to_csv(out / "adm_final_probability.csv", index=False)
    np.savez_compressed(out / "stage2_optimizer_state.npz", best_position=adm.best_decision_vector,
                        best_species=adm.best_strategies, probability_history=np.asarray(adm.probability_history))
    save_json(out / "stage2_convergence.json", adm.convergence)
    manifest.update(status="complete", stage1_verified=True, stage2_global_rollback=True,
                    final_conflict_points=len(final), final_conflict_pairs=count_conflict_pairs(final))
    save_json(out / "manifest.json", manifest)
    comparison = ["final_conflict_points", "final_conflict_pairs", "final_exact_survivors", "final_moved_same_pair",
                  "final_new_conflict_points", "risk_increase_percent", "delayed_flights", "changed_flights", "stage2_runtime_sec"]
    table = "\n".join(f"| {key} | {old2[key]} | {m2[key]} |" for key in comparison)
    selected_counts = m2["selected_plan_action_counts"]
    report = f"""# Stage2 Global Rollback Diagnostic

This is a diagnostic experiment, NOT a claim that the paper contains rollback.
Branch: scheduling. Frozen source snapshot: `{SOURCE}`. Baseline hash: `{HASH}`.
Only Stage2 candidate local-action acceptance changed. No repair, new candidates, seeds, or parameter tuning.
ADM and FATA ran for 200 generations with NP=50, Parf=0.2. Optimizer base seed=0;
Stage2 RNG seed=206 is the unchanged existing scheduler offset, not an extra seed search.

| Metric | NO ROLLBACK | WITH ROLLBACK |
| --- | --- | --- |
{table}

1. Stage1 strictly reproduced 130 -> 69, 60 pairs, identical fitness {stage1.fitness}; Top10 `{keys}`. No Stage1 modifications.
2. Final: 69 -> {len(final)} points, {count_conflict_pairs(final)} pairs.
3. Zero conflict: {not final}.
4. New-pair points: 5 -> {m2['final_new_conflict_points']}. Original new flight pairs still present: {m2['original_new_pairs_still_present']}. This checks pairs, not merely aggregate counts.
5. Search rollback count={m2['rollback_count']}; accepted actions={m2['accepted_action_count']}. Pair increase rejects={m2['pairs_increased_count']}; equal pairs with non-decreased points rejects={m2['same_pairs_points_not_decreased_count']}.
6. Exact survivors: 4 -> {m2['final_exact_survivors']}; moved same pair: 0 -> {m2['final_moved_same_pair']}; new points: 5 -> {m2['final_new_conflict_points']}. Old-conflict increase with fewer new conflicts: {m2['final_exact_survivors'] + m2['final_moved_same_pair'] > 4 and m2['final_new_conflict_points'] < 5}.
7. Stage2 time increased by {m2['stage2_runtime_increase_sec']:.3f}s ({m2['stage2_runtime_ratio']:.3f}x); baseline Stage1={old1['stage1_runtime_sec']:.3f}s, rerun Stage1={seconds1:.3f}s. Timing includes validation/logging overhead and machine variability.
8. Risk change={m2['risk_increase_percent']:.6f}%; worse than no rollback: {m2['risk_increase_percent'] > old2['risk_increase_percent']}.
9. Delayed flights: 28 -> {m2['delayed_flights']}; increased: {m2['delayed_flights'] > 28}. Changed flights: 50 -> {m2['changed_flights']}.
10. This single-scene/single-seed diagnostic cannot establish lack of global protection as the main cause. It demonstrates the observed trade-off above. A strict lexicographic improvement can still create a new pair if more old pairs disappear; suppression of every new pair is NOT guaranteed.

## Action Semantics And Coordination Limitation

Each candidate starts from the same Stage1 schedule. Actions follow unchanged block order and schedule/speed/reroute order.
Existing shared per-flight ATD and selected speed segments / merged route windows are applied together per flight x strategy,
with all incident conflict indices logged. No duplicate conflicting assignments or extra decision variables were introduced.
All 100 flights are fully detected before and after EACH action; equal conflict counts are rejected even if fitness improves.
Infeasible route candidates retain the original infinite-fitness rule and are not converted into repair candidates.
Rollback restores the exact prior actor reference; other flight plans are immutable during that action.
Rejecting one action can prevent a later cooperative combination from improving. The experiment deliberately retains this side effect.
ADM still learns sampled species without filtering rejected strategy labels; no acceptance-adaptive probability update was added.

Search counts include worker candidate evaluations, incumbent reevaluations, and ADM's final decode, including attempted actions
in candidates that later become geometrically infeasible. They are NOT unique actions committed to the final schedule.
Final selected-plan replay counts: `{selected_counts}`; detailed trace: `stage2_selected_action_log.csv`.
Worker logs are retained in `search_actions/`; aggregate full trace is `stage2_action_log.csv`; rejected actions are `stage2_rollback_log.csv`.
The final replay is outside measured optimizer runtime and agrees exactly with ADM's returned fitness.
"""
    (out / "stage2_rollback_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(m2, indent=2), flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-run", type=Path)
    args = parser.parse_args()
    if args.audit_run:
        search = pd.read_csv(args.audit_run / "stage2_action_log.csv")
        selected = pd.read_csv(args.audit_run / "stage2_selected_action_log.csv")
        result = dict(search=audit_actions(search), selected=audit_actions(selected))
        baseline_bytes = (args.audit_run / "baseline_initial_plans.pkl").read_bytes()
        assert hashlib.sha256(baseline_bytes).hexdigest() == HASH
        baseline_plans = pickle.loads(baseline_bytes)
        cfg = load_config(args.audit_run / "baseline_config.yaml")
        cfg["conflict"]["t_conflict"] = 30.0
        grid = AirspaceGrid.from_config(cfg, seed=cfg.get("environment_seed", 2025))
        final_plans = pickle.loads((args.audit_run / "final_plans.pkl").read_bytes())
        assert len(final_plans) == 100
        assert all(_valid_route(p.path, p, grid) for p in final_plans)
        conflicts = detect_conflicts(final_plans, cfg, uncertain=True)
        expected = pd.read_csv(args.audit_run / "final_conflicts.csv")
        actual = pd.DataFrame([_conflict_row(c) for c in conflicts], columns=expected.columns)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-12, atol=1e-12)
        initial = detect_conflicts(baseline_plans, cfg, uncertain=True)
        assert (len(initial), count_conflict_pairs(initial)) == (130, 85)
        expected_provenance = pd.read_csv(args.audit_run / "final_conflict_provenance.csv")
        pd.testing.assert_frame_equal(provenance(initial, conflicts), expected_provenance, check_dtype=False)
        result["final_global_recheck"] = dict(flight_count=100, valid_routes=True, baseline_hash_verified=True,
                                             conflict_points=len(conflicts), conflict_pairs=count_conflict_pairs(conflicts),
                                             full_conflict_records_match=True, provenance_match=True)
        (args.audit_run / "no_rollback_final_conflict_provenance.csv").write_bytes(
            source_file(f"outputs/multiline_baseline_full_run/{BASELINE_RUN}/final_conflict_provenance.csv"))
        save_json(args.audit_run / "action_audit.json", result)
        print(json.dumps(result, indent=2))
    else:
        main()
