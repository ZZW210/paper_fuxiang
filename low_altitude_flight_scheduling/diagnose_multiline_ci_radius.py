"""Diagnose Multi-degree collective influence radii on the frozen baseline only."""
from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

from src.baseline_plans import baseline_metadata, baseline_plan_hash, load_baseline_plans
from src.config import load_config
from src.conflict_detection import count_conflict_pairs, detect_conflicts
from src.scene_diagnostics import coverage


L_VALUES = (1, 2, 3, 4)
DENSE_IDS = (52, 77, 70, 87)
EXPECTED = (99, 130, 85)


def _pair(conflict):
    return tuple(sorted((int(conflict.plan_a), int(conflict.plan_b))))


def build_networks(plans, conflicts):
    topology, multi = nx.Graph(), nx.MultiGraph()
    topology.add_nodes_from(plan.id for plan in plans)
    multi.add_nodes_from(plan.id for plan in plans)
    for index, conflict in enumerate(conflicts):
        a, b = int(conflict.plan_a), int(conflict.plan_b)
        topology.add_edge(a, b)
        multi.add_edge(a, b, conflict_index=index)
    return topology, multi


def radius_scores(topology, multi, l_values=L_VALUES):
    degrees = dict(multi.degree())
    scores, shells = {}, {}
    for radius in l_values:
        score, shell_map = {}, {}
        for node in topology.nodes:
            shell = sorted(
                other for other, distance in nx.single_source_shortest_path_length(topology, node, cutoff=radius).items()
                if distance == radius
            )
            shell_map[node] = shell
            score[node] = float((degrees[node] - 1) * sum(max(0, degrees[other] - 1) for other in shell))
        scores[radius], shells[radius] = score, shell_map
    return degrees, scores, shells


def _rank(scores):
    ordered = sorted(scores, key=lambda flight_id: (-scores[flight_id], flight_id))
    return ordered, {flight_id: index + 1 for index, flight_id in enumerate(ordered)}


def _top10_summary(conflicts, ids, degrees, partners):
    selected = set(ids)
    covered = [conflict for conflict in conflicts if conflict.plan_a in selected or conflict.plan_b in selected]
    all_pairs = {_pair(conflict) for conflict in conflicts}
    covered_pairs = {_pair(conflict) for conflict in covered}
    endpoint_incidence = sum(degrees[flight_id] for flight_id in ids)
    unique_records = len(covered)
    redundant = endpoint_incidence - unique_records
    return dict(
        covered_conflict_points=unique_records,
        coverage_ratio=unique_records / len(conflicts),
        covered_conflict_pairs=len(covered_pairs),
        unique_pair_coverage=len(covered_pairs) / len(all_pairs),
        mean_multi_degree=float(sum(degrees[flight_id] for flight_id in ids) / len(ids)),
        median_multi_degree=float(pd.Series([degrees[flight_id] for flight_id in ids]).median()),
        mean_unique_partners=float(sum(partners[flight_id] for flight_id in ids) / len(ids)),
        median_unique_partners=float(pd.Series([partners[flight_id] for flight_id in ids]).median()),
        unique_involved_flights=len({flight_id for conflict in covered for flight_id in (conflict.plan_a, conflict.plan_b)}),
        endpoint_incidence_count=endpoint_incidence,
        unique_covered_conflict_records=unique_records,
        redundant_endpoint_incidence=redundant,
        redundancy_ratio=redundant / endpoint_incidence if endpoint_incidence else 0.0,
    )


def _attack(topology, multi, ranked, radius):
    simple, multi_graph = topology.copy(), multi.copy()
    rows = []
    for removed in range(11):
        active = [node for node, degree in simple.degree() if degree > 0]
        component_size = max((len(component) for component in nx.connected_components(simple)), default=0)
        rows.append(dict(l=radius, removed_k=removed, remaining_unique_edges=simple.number_of_edges(),
                         remaining_multi_edges=multi_graph.number_of_edges(), remaining_connected_nodes=len(active),
                         largest_component_nodes=component_size))
        if removed < 10:
            node = ranked[removed]
            simple.remove_node(node)
            multi_graph.remove_node(node)
    return rows


def _plot_rank(dense, output):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for flight_id, group in dense.groupby("flight_id"):
        ax.plot(group.l, group[[f"rank_l{int(value)}" for value in group.l]].to_numpy().diagonal() if False else group["rank"], marker="o", label=str(flight_id))
    ax.set(xticks=list(L_VALUES), xlabel="CI shell radius l", ylabel="CI rank (lower is higher)", title="Dense-flight Multi-CI rank by radius")
    ax.invert_yaxis(); ax.legend(title="flight_id"); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def _plot_coverage(summary, output):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(summary.l, summary.coverage_ratio * 100, marker="o", color="#3366AA")
    ax.set(xticks=list(L_VALUES), xlabel="CI shell radius l", ylabel="Top10 conflict-point coverage (%)", title="Multi-CI Top10 coverage by radius")
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def _plot_attack(attack, output):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for radius, group in attack.groupby("l"):
        ax.plot(group.removed_k, group.largest_component_nodes, marker="o", label=f"l={radius}")
    ax.set(xlabel="Top-ranked nodes removed", ylabel="Largest connected component nodes", title="Multi-CI network attack diagnostic")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output, dpi=180); plt.close(fig)


def main():
    root = Path(__file__).resolve().parent
    baseline = baseline_metadata(root)
    if baseline_plan_hash(root) != baseline["sha256"]:
        raise AssertionError("Frozen baseline hash mismatch")
    cfg = load_config(root / "config.yaml", {"optimization": {"scheduler_mode": "paper_strict"}})
    cfg["conflict"]["t_conflict"] = 30.0
    plans = load_baseline_plans(root)
    deterministic = detect_conflicts(plans, cfg, uncertain=False)
    conflicts = detect_conflicts(plans, cfg, uncertain=True)
    if (len(deterministic), len(conflicts), count_conflict_pairs(conflicts)) != EXPECTED:
        raise RuntimeError(f"Baseline conflicts changed: {(len(deterministic), len(conflicts), count_conflict_pairs(conflicts))}")
    topology, multi = build_networks(plans, conflicts)
    degrees, scores, shells = radius_scores(topology, multi)
    partners = dict(topology.degree())
    involvement = Counter(fid for conflict in conflicts for fid in (conflict.plan_a, conflict.plan_b))
    if any(degrees[plan.id] != involvement[plan.id] for plan in plans):
        raise AssertionError("Multi degree does not equal incident conflict records")
    rankings = {radius: _rank(scores[radius]) for radius in L_VALUES}
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    run_id = f"{stamp}_baseline_multidegree_ci_radius"
    out = root / "outputs" / "ci_radius_baseline_diagnosis" / run_id
    out.mkdir(parents=True)

    node_rows = []
    for plan in plans:
        row = dict(flight_id=plan.id, multi_degree=degrees[plan.id], unique_conflict_partners=partners[plan.id], conflict_point_involvement=involvement[plan.id])
        for radius in L_VALUES:
            row[f"CI_l{radius}"] = scores[radius][plan.id]
            row[f"rank_l{radius}"] = rankings[radius][1][plan.id]
        node_rows.append(row)
    all_nodes = pd.DataFrame(node_rows).sort_values("flight_id")
    all_nodes.to_csv(out / "ci_radius_all_nodes.csv", index=False)

    top_rows, summary_rows, attack_rows, dense_rows, audit_lines = [], [], [], [], ["# Dense Flight Multi-CI Shell Audit", ""]
    for radius in L_VALUES:
        ranked = rankings[radius][0]
        top10 = ranked[:10]
        values = _top10_summary(conflicts, top10, degrees, partners)
        summary_rows.append(dict(l=radius, top10=json.dumps(top10), **values))
        attack_rows.extend(_attack(topology, multi, ranked, radius))
        for rank, flight_id in enumerate(ranked[:20], 1):
            top_rows.append(dict(l=radius, rank=rank, flight_id=flight_id, multi_degree=degrees[flight_id], unique_conflict_partners=partners[flight_id], conflict_point_involvement=involvement[flight_id], CI=scores[radius][flight_id]))
    for flight_id in DENSE_IDS:
        audit_lines += [f"## Flight {flight_id}", f"D_multi({flight_id}) = {degrees[flight_id]}", ""]
        row = dict(flight_id=flight_id, multi_degree=degrees[flight_id], unique_conflict_partners=partners[flight_id], conflict_point_involvement=involvement[flight_id])
        for radius in L_VALUES:
            shell = shells[radius][flight_id]
            details = [dict(flight_id=node, multi_degree=degrees[node], degree_minus_one=max(0, degrees[node] - 1)) for node in shell]
            degree_sum = sum(item["degree_minus_one"] for item in details)
            row.update({f"shell_l{radius}_nodes": json.dumps(shell), f"shell_l{radius}_degree_sum": degree_sum,
                        f"shell_l{radius}_details": json.dumps(details), f"CI_l{radius}": scores[radius][flight_id],
                        f"rank_l{radius}": rankings[radius][1][flight_id]})
            audit_lines += [f"### l={radius}", f"shell_{radius}({flight_id}) = {shell}",
                            f"D_multi(v)-1 values = {json.dumps(details)}", f"sum(D_multi(v)-1) = {degree_sum}",
                            f"CI_{radius}({flight_id}) = ({degrees[flight_id]} - 1) * {degree_sum} = {scores[radius][flight_id]}", ""]
        dense_rows.append(row)
        print(f"{flight_id}: " + ", ".join(f"l={radius} CI={scores[radius][flight_id]} rank={rankings[radius][1][flight_id]}" for radius in L_VALUES), flush=True)
    dense = pd.DataFrame(dense_rows)
    top20, summary, attack = pd.DataFrame(top_rows), pd.DataFrame(summary_rows), pd.DataFrame(attack_rows)
    dense.to_csv(out / "ci_radius_dense_flights.csv", index=False)
    top20.to_csv(out / "ci_radius_top20.csv", index=False)
    summary.to_csv(out / "ci_radius_top10_summary.csv", index=False)
    attack.to_csv(out / "ci_radius_attack.csv", index=False)
    (out / "ci_radius_dense_audit.md").write_text("\n".join(audit_lines), encoding="utf-8")

    rank_plot = pd.concat([pd.DataFrame(dict(flight_id=flight_id, l=list(L_VALUES), rank=[rankings[radius][1][flight_id] for radius in L_VALUES])) for flight_id in DENSE_IDS], ignore_index=True)
    _plot_rank(rank_plot, out / "ci_radius_rank_comparison.png")
    _plot_coverage(summary, out / "ci_radius_coverage.png")
    _plot_attack(attack, out / "ci_radius_attack_lcc.png")
    best_coverage = summary[summary.coverage_ratio.eq(summary.coverage_ratio.max())].l.tolist()
    best_degree = summary[summary.mean_multi_degree.eq(summary.mean_multi_degree.max())].l.tolist()
    best_partners = summary[summary.mean_unique_partners.eq(summary.mean_unique_partners.max())].l.tolist()
    highest_redundancy = summary[summary.redundancy_ratio.eq(summary.redundancy_ratio.max())].l.tolist()
    attack10 = attack[attack.removed_k.eq(10)]
    strongest_attack = attack10[attack10.largest_component_nodes.eq(attack10.largest_component_nodes.min())].l.tolist()
    tops = {radius: rankings[radius][0][:10] for radius in L_VALUES}
    report = f"""# Multi-degree CI Radius Diagnosis

Frozen baseline hash: `{baseline['sha256']}`. No generator, optimizer, Stage1, Stage2, or FATA was run.

- Initial deterministic / uncertain conflicts / unique pairs: {len(deterministic)} / {len(conflicts)} / {count_conflict_pairs(conflicts)}.
- Top10 l=1: `{tops[1]}`
- Top10 l=2: `{tops[2]}`
- Top10 l=3: `{tops[3]}`
- Top10 l=4: `{tops[4]}`

## Dense flights

The full manual shell calculation is in `ci_radius_dense_audit.md`; compact values are in `ci_radius_dense_flights.csv`.

{chr(10).join('- Flight ' + str(row.flight_id) + ': ' + ', '.join('l=' + str(radius) + ' CI=' + str(row[f'CI_l{radius}']) + ' rank=' + str(int(row[f'rank_l{radius}'])) for radius in L_VALUES) for _, row in dense.iterrows())}

## Comparative evidence

- Highest Top10 conflict-point coverage: l={best_coverage}; this is a record-level count with every covered conflict counted once.
- Highest mean Multi-degree among Top10: l={best_degree}.
- Highest mean unique conflict partners among Top10: l={best_partners}.
- Most repeated endpoint selection: l={highest_redundancy}; its redundancy ratio is reported in `ci_radius_top10_summary.csv` and is not a coverage gain.
- Strongest Top10 removal by final LCC size: l={strongest_attack}; the attack diagnostic does not distinguish these radii when tied.

l=2 can give a high-Multi-degree node CI=0 because the formula requires a nonzero sum over the exact simple-topology two-hop shell. Repeated parallel conflicts increase D_multi but do not create additional shell nodes. Here, flight 52 has only flight 76 in its l=2 shell and D_multi(76)-1=0; the full arithmetic is in the dense audit.

Conclusion: l=1 most directly reflects the repeated-conflict dense pairs in this frozen baseline and has the highest Top10 point coverage, while l=2 has the highest Top10 partner diversity and lower redundancy. {('The metrics conflict, so this baseline cannot uniquely choose l or establish the paper\'s undisclosed radius.' if len(set(best_coverage + best_degree + best_partners + strongest_attack)) > 1 else 'One radius is consistently strongest in these diagnostics, but that remains a baseline-specific interpretation.')}
"""
    (out / "ci_radius_report.md").write_text(report, encoding="utf-8")
    manifest = dict(baseline_plan_hash=baseline["sha256"], deterministic_conflicts=len(deterministic), uncertain_conflicts=len(conflicts),
                    unique_pairs=count_conflict_pairs(conflicts), degree_mode="multi", tested_l=list(L_VALUES),
                    **{f"Top10_l{radius}": tops[radius] for radius in L_VALUES},
                    **{f"coverage_l{radius}": float(summary.loc[summary.l.eq(radius), "coverage_ratio"].iloc[0]) for radius in L_VALUES},
                    optimizers_executed=False)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
