"""Diagnose binary, multigraph, and weighted initial conflict-network representations."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from src.config import load_config, resolve_scene_seeds
from src.conflict_detection import detect_conflicts, group_continuous_conflicts
from src.flight_plan import plan_paper_random_traffic, sample_paper_random_traffic
from src.grid import AirspaceGrid
from src.risk_map import generate_risk_map
from src.scene_diagnostics import degree_gini

L_VALUES = (1, 2, 3, 4)
PAPER_COVERAGE = 78 / 97


def pair_key(conflict):
    return tuple(sorted((int(conflict.plan_a), int(conflict.plan_b))))


def shell_ci(topology: nx.Graph, values: dict[int, float], l: int) -> dict[int, float]:
    """CI values with an externally supplied degree/strength and simple-graph shells."""
    scores = {}
    for node in topology.nodes:
        shell = [other for other, distance in nx.single_source_shortest_path_length(topology, node, cutoff=l).items() if distance == l]
        scores[node] = float((values[node] - 1) * sum(max(0.0, values[other] - 1) for other in shell))
    return scores


def build_graphs(plan_ids, conflicts):
    binary, multi = nx.Graph(), nx.MultiGraph()
    binary.add_nodes_from(plan_ids); multi.add_nodes_from(plan_ids)
    counts = Counter(pair_key(conflict) for conflict in conflicts)
    for conflict in conflicts:
        multi.add_edge(conflict.plan_a, conflict.plan_b)
    for pair, count in counts.items():
        binary.add_edge(*pair, weight=count)
    return binary, multi, counts


def node_metrics(plan_ids, conflicts, binary, multi) -> tuple[pd.DataFrame, dict]:
    involvement = Counter()
    for conflict in conflicts:
        involvement.update((conflict.plan_a, conflict.plan_b))
    binary_degree = dict(binary.degree())
    multi_degree = dict(multi.degree())
    strength = dict(binary.degree(weight="weight"))
    rows = []
    scores = {"binary": {}, "multi": {}, "weighted": {}}
    for l in L_VALUES:
        scores["binary"][l] = shell_ci(binary, binary_degree, l)
        scores["multi"][l] = shell_ci(binary, multi_degree, l)
        scores["weighted"][l] = shell_ci(binary, strength, l)
    pagerank = nx.pagerank(binary) if binary.number_of_edges() else dict.fromkeys(plan_ids, 0.0)
    betweenness = nx.betweenness_centrality(binary) if binary.number_of_edges() else dict.fromkeys(plan_ids, 0.0)
    for fid in plan_ids:
        row = dict(flight_id=fid, unique_conflict_partners=int(binary_degree[fid]), conflict_point_involvement=int(involvement[fid]),
                   binary_degree=int(binary_degree[fid]), multi_degree=int(multi_degree[fid]), weighted_strength=int(strength[fid]),
                   pagerank_binary=float(pagerank[fid]), betweenness_binary=float(betweenness[fid]))
        for name in scores:
            for l in L_VALUES:
                row[f"CI_{name}_l{l}"] = scores[name][l][fid]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("flight_id"), scores


def pair_distribution(conflicts) -> pd.DataFrame:
    by_pair = {}
    for conflict in conflicts:
        by_pair.setdefault(pair_key(conflict), []).append(conflict)
    segments = Counter(); max_len = Counter()
    for segment in group_continuous_conflicts(conflicts):
        pair = (segment.plan_a, segment.plan_b)
        segments[pair] += 1; max_len[pair] = max(max_len[pair], len(segment.conflicts))
    rows = []
    for pair, group in by_pair.items():
        ordered = sorted(group, key=lambda item: (item.idx_a, item.idx_b, item.time_a))
        rows.append(dict(flight_a=pair[0], flight_b=pair[1], conflict_point_count=len(group),
                         continuous_segment_count=segments[pair], max_continuous_segment_length=max_len[pair],
                         first_conflict_cell="(" + ",".join(map(str, ordered[0].cell)) + ")",
                         last_conflict_cell="(" + ",".join(map(str, ordered[-1].cell)) + ")",
                         time_gap_mean=float(np.mean([item.time_gap for item in group])),
                         time_gap_min=float(np.min([item.time_gap for item in group])),
                         time_gap_max=float(np.max([item.time_gap for item in group]))))
    return pd.DataFrame(rows).sort_values(["conflict_point_count", "flight_a", "flight_b"], ascending=[False, True, True])


def top10_outputs(metrics: pd.DataFrame, conflicts) -> tuple[pd.DataFrame, pd.DataFrame]:
    pair_total = len({pair_key(conflict) for conflict in conflicts})
    rows, summaries = [], []
    for network_type in ("binary", "multi", "weighted"):
        for l in L_VALUES:
            score_name = f"CI_{network_type}_l{l}"
            ranked = metrics.sort_values([score_name, "flight_id"], ascending=[False, True], kind="stable").head(10)
            ids = set(ranked.flight_id)
            covered = [conflict for conflict in conflicts if conflict.plan_a in ids or conflict.plan_b in ids]
            covered_pairs = {pair_key(conflict) for conflict in covered}
            incident = metrics[metrics.flight_id.isin(ids)]
            summaries.append(dict(network_type=network_type, l=l,
                                  conflict_point_coverage=len(covered) / len(conflicts),
                                  conflict_pair_coverage=len(covered_pairs) / pair_total if pair_total else 0.0,
                                  covered_unique_flights=len({fid for conflict in covered for fid in (conflict.plan_a, conflict.plan_b)}),
                                  top10_total_incident_conflict_points=int(incident.conflict_point_involvement.sum()),
                                  top10_mean_incident_conflict_points=float(incident.conflict_point_involvement.mean()),
                                  top10_median_incident_conflict_points=float(incident.conflict_point_involvement.median())))
            for rank, item in enumerate(ranked.itertuples(index=False), 1):
                rows.append(dict(network_type=network_type, l=l, rank=rank, flight_id=item.flight_id,
                                 score=getattr(item, score_name), unique_conflict_partners=item.unique_conflict_partners,
                                 conflict_point_involvement=item.conflict_point_involvement))
    return pd.DataFrame(rows), pd.DataFrame(summaries)


def _network_stats(binary, multi, metrics) -> pd.DataFrame:
    base = dict(node_count=binary.number_of_nodes(), unique_pair_count=binary.number_of_edges(), edge_count=binary.number_of_edges(),
                multi_edge_count=multi.number_of_edges(), weighted_edge_sum=sum(dict(binary.degree(weight="weight")).values()) / 2,
                mean_binary_degree=float(metrics.binary_degree.mean()), max_binary_degree=int(metrics.binary_degree.max()),
                mean_multi_degree=float(metrics.multi_degree.mean()), max_multi_degree=int(metrics.multi_degree.max()),
                mean_strength=float(metrics.weighted_strength.mean()), max_strength=int(metrics.weighted_strength.max()),
                degree_gini=degree_gini(metrics.binary_degree), multi_degree_gini=degree_gini(metrics.multi_degree), strength_gini=degree_gini(metrics.weighted_strength))
    return pd.DataFrame([dict(network_type=name, **base) for name in ("binary", "multi", "weighted")])


def _draw_network(graph, pos, values, path, title, weighted=False, multi=False):
    fig, ax = plt.subplots(figsize=(9, 7.5)); maximum = max(1.0, max(values.values()))
    if multi:
        grouped = Counter(tuple(sorted((u, v))) for u, v, _ in graph.edges(keys=True))
        for (u, v), count in grouped.items():
            for index in range(min(count, 12)):
                rad = (index - (min(count, 12) - 1) / 2) * .045
                nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], connectionstyle=f"arc3,rad={rad}", alpha=.18, width=.7, ax=ax)
        note = "Visual parallel edges capped at 12 per pair; node degree uses all conflict points."
    else:
        widths = [0.6 + 3.4 * data.get("weight", 1) / max(1, max(dict(graph.degree(weight="weight")).values())) for _, _, data in graph.edges(data=True)] if weighted else .7
        nx.draw_networkx_edges(graph, pos, width=widths, alpha=.28, ax=ax); note = "Edge width encodes conflict-point count." if weighted else "One edge per unique conflicting flight pair."
    nodes = nx.draw_networkx_nodes(graph, pos, node_color=[values[node] for node in graph.nodes], node_size=85, cmap="YlOrRd", vmin=0, vmax=maximum, ax=ax)
    ax.set_title(title); ax.axis("off"); fig.colorbar(nodes, ax=ax, shrink=.75, label="Node metric")
    fig.text(.01, .01, note, fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _paper_style(binary, multi, positions, metrics, path):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharex=True, sharey=True)
    values = [dict(zip(metrics.flight_id, metrics.binary_degree)), dict(zip(metrics.flight_id, metrics.multi_degree)), dict(zip(metrics.flight_id, metrics.weighted_strength))]
    max_value = max(max(value.values()) for value in values)
    for ax, graph, value, name in zip(axes, (binary, multi, binary), values, ("binary", "multigraph", "weighted")):
        if name == "multigraph":
            grouped = Counter(tuple(sorted((u, v))) for u, v, _ in multi.edges(keys=True))
            for (u, v), count in grouped.items():
                for index in range(min(count, 12)):
                    nx.draw_networkx_edges(multi, positions, edgelist=[(u, v)], connectionstyle=f"arc3,rad={(index-(min(count,12)-1)/2)*.04}", alpha=.15, width=.6, ax=ax)
        else:
            widths = [0.6 + 3.2 * data.get("weight", 1) / max_value for _, _, data in binary.edges(data=True)] if name == "weighted" else .6
            nx.draw_networkx_edges(binary, positions, width=widths, alpha=.22, ax=ax)
        nodes = nx.draw_networkx_nodes(graph, positions, node_size=58, node_color=[value[node] for node in graph.nodes], cmap="YlOrRd", vmin=0, vmax=max_value, ax=ax)
        ax.set_title(name); ax.set_aspect("equal"); ax.set_xlabel("x midpoint (cell)")
    axes[0].set_ylabel("y midpoint (cell)"); fig.colorbar(nodes, ax=axes, shrink=.75, label="degree / strength (common scale)")
    fig.suptitle("Paper-style fixed midpoint layout; multigraph visual edges capped at 12 per pair")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _report(out, distribution, metrics, coverage):
    top = distribution.head(5); total = 107
    best_by_type = coverage.sort_values(["network_type", "conflict_point_coverage"], ascending=[True, False]).groupby("network_type", as_index=False).first()
    corr = {name: metrics[name].corr(metrics.conflict_point_involvement, method="spearman") for name in ["binary_degree", "multi_degree", "weighted_strength", *[f"CI_{kind}_l{l}" for kind in ("binary", "multi", "weighted") for l in L_VALUES]]}
    highest = max(corr, key=lambda key: abs(corr[key]))
    def rank(fid, column):
        return int(metrics.sort_values([column, "flight_id"], ascending=[False, True], kind="stable").reset_index(drop=True).query("flight_id == @fid").index[0] + 1)
    multi73 = [rank(73, f"CI_multi_l{l}") for l in L_VALUES]; weighted73 = [rank(73, f"CI_weighted_l{l}") for l in L_VALUES]
    binary98 = [rank(98, f"CI_binary_l{l}") for l in L_VALUES]; multi98 = [rank(98, f"CI_multi_l{l}") for l in L_VALUES]
    text = f"""# Conflict Network Representation Diagnosis

Fixed scene: environment seed 2025, traffic seed 316, deterministic conflicts=53, uncertain conflicts=107. No scheduling or optimization stage was run.

## Repeated conflict pairs

Top1/Top2/Top5 pairs contribute {int(top.iloc[0].conflict_point_count)}/{total} ({top.iloc[0].conflict_point_count/total:.2%}), {int(top.head(2).conflict_point_count.sum())}/{total} ({top.head(2).conflict_point_count.sum()/total:.2%}), and {int(top.conflict_point_count.sum())}/{total} ({top.conflict_point_count.sum()/total:.2%}) conflict points.

## Findings

A. The paper's textual e_uv in {{0,1}} definition is closest to the binary simple graph.

B. The fixed-midpoint visual behavior with repeated parallel edges is closest to the multigraph; weighted simple edges retain multiplicity only through width.

C. Binary CI can understate repeated-conflict importance because its degree counts unique partners, not repeated conflict points.

D. Best Top10 point coverage by representation: {best_by_type.to_dict('records')} (paper reference: {PAPER_COVERAGE:.4%}, diagnostic only).

E. The metric most correlated with incident conflict-point involvement is `{highest}` (Spearman={corr[highest]:.6f}). Multi degree and weighted strength are expected to be highly aligned with that involvement by construction.

F. Flight 73 multi-CI ranks l1..l4: {multi73}; weighted-CI ranks: {weighted73}. Flight 98 binary-CI ranks: {binary98}; multi-CI ranks: {multi98}.

The evidence supports a tension between the textual binary adjacency definition and plotted network-degree behavior: **evidence suggests a discrepancy between the textual binary adjacency definition and the plotted network-degree behavior.** It does not prove the paper used a MultiGraph. Figure 8 behavior may be better explained by a multi/weighted representation because repeated edges and conflict multiplicity become visible.
"""
    (out / "conflict_network_representation_report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment-seed", type=int, default=2025); parser.add_argument("--traffic-seed", type=int, default=316)
    args = parser.parse_args(); root = Path(__file__).resolve().parent
    cfg = load_config(root / "config.yaml", {"optimization": {"scheduler_mode": "paper_strict"}})
    resolve_scene_seeds(cfg, environment_seed=args.environment_seed, traffic_seed=args.traffic_seed)
    grid = AirspaceGrid.from_config(cfg, seed=args.environment_seed); risk = generate_risk_map(grid, cfg)
    plans = plan_paper_random_traffic(grid, risk, cfg, sample_paper_random_traffic(grid, cfg, 100, args.traffic_seed))
    deterministic, conflicts = detect_conflicts(plans, cfg, uncertain=False), detect_conflicts(plans, cfg, uncertain=True)
    if (len(deterministic), len(conflicts)) != (53, 107): raise RuntimeError(f"expected fixed 53/107, got {len(deterministic)}/{len(conflicts)}")
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")[:-3]; out = root / "outputs" / "conflict_network_representation" / f"{stamp}_env2025_traffic316"; out.mkdir(parents=True)
    binary, multi, _ = build_graphs([plan.id for plan in plans], conflicts); metrics, _ = node_metrics([plan.id for plan in plans], conflicts, binary, multi)
    distribution = pair_distribution(conflicts); top = distribution.head(20).copy(); top.insert(0, "rank", range(1, len(top)+1)); top["binary_edge_count"] = 1; top["multi_edge_count"] = top.conflict_point_count; top["edge_weight"] = top.conflict_point_count
    comparison, coverage = top10_outputs(metrics, conflicts)
    corr_rows = [dict(metric=column, spearman_correlation=float(metrics[column].corr(metrics.conflict_point_involvement, method="spearman"))) for column in metrics.columns if column not in {"flight_id", "conflict_point_involvement"}]
    pairs = Counter(pair_key(conflict) for conflict in conflicts); involvement = Counter(fid for conflict in conflicts for fid in (conflict.plan_a, conflict.plan_b))
    high_ids = set(metrics.nlargest(10, "multi_degree").flight_id) | set(metrics.nlargest(10, "weighted_strength").flight_id)
    concentration = pd.DataFrame([dict(flight_id=fid, multi_degree=int(metrics.loc[metrics.flight_id.eq(fid), "multi_degree"].iloc[0]), weighted_strength=int(metrics.loc[metrics.flight_id.eq(fid), "weighted_strength"].iloc[0]), max_pair_conflict_points=max((count for pair, count in pairs.items() if fid in pair), default=0), max_pair_share=max((count for pair, count in pairs.items() if fid in pair), default=0) / involvement[fid] if involvement[fid] else 0.0) for fid in sorted(high_ids)])
    case_rows=[]
    for fid in (49,73,98):
        base=metrics[metrics.flight_id.eq(fid)].iloc[0]
        for kind in ("binary","multi","weighted"):
            for l in L_VALUES:
                column=f"CI_{kind}_l{l}"; rank=int(metrics.sort_values([column,"flight_id"],ascending=[False,True],kind="stable").reset_index(drop=True).query("flight_id == @fid").index[0]+1)
                case_rows.append(dict(flight_id=fid, network_type=kind, l=l, unique_conflict_partners=base.unique_conflict_partners, conflict_point_involvement=base.conflict_point_involvement, binary_degree=base.binary_degree, multi_degree=base.multi_degree, weighted_strength=base.weighted_strength, CI=base[column], rank=rank))
    distribution.to_csv(out/"conflict_pair_distribution.csv",index=False); top.to_csv(out/"top_repeated_conflict_pairs.csv",index=False); metrics.to_csv(out/"node_metrics_all.csv",index=False); pd.DataFrame(corr_rows).to_csv(out/"metric_correlation.csv",index=False); comparison.to_csv(out/"top10_comparison.csv",index=False); coverage.to_csv(out/"top10_coverage_summary.csv",index=False); _network_stats(binary,multi,metrics).to_csv(out/"network_statistics.csv",index=False); concentration.to_csv(out/"high_degree_pair_concentration.csv",index=False); pd.DataFrame(case_rows).to_csv(out/"case_study_49_73_98.csv",index=False)
    spring=nx.spring_layout(binary,seed=7); _draw_network(binary,spring,dict(zip(metrics.flight_id,metrics.binary_degree)),out/"conflict_network_binary.png","Binary simple graph: binary degree"); _draw_network(multi,spring,dict(zip(metrics.flight_id,metrics.multi_degree)),out/"conflict_network_multigraph.png","Multigraph: degree counts all parallel edges",multi=True); _draw_network(binary,spring,dict(zip(metrics.flight_id,metrics.weighted_strength)),out/"conflict_network_weighted.png","Weighted graph: node strength",weighted=True)
    midpoint={plan.id:((plan.start[0]+plan.goal[0])/2,(plan.start[1]+plan.goal[1])/2) for plan in plans}; _paper_style(binary,multi,midpoint,metrics,out/"paper_style_network_comparison.png")
    (out/"manifest.json").write_text(json.dumps(dict(environment_seed=args.environment_seed,traffic_seed=args.traffic_seed,initial_deterministic_conflicts=len(deterministic),initial_uncertain_conflicts=len(conflicts),stages_executed=[],representations=["binary","multi","weighted"],l_values=L_VALUES),indent=2),encoding="utf-8"); _report(out,distribution,metrics,coverage)
    print(out)


if __name__ == "__main__": main()
