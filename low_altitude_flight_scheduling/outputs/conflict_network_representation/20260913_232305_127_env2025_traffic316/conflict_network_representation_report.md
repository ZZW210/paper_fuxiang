# Conflict Network Representation Diagnosis

Fixed scene: environment seed 2025, traffic seed 316, deterministic conflicts=53, uncertain conflicts=107. No scheduling or optimization stage was run.

## Repeated conflict pairs

Top1/Top2/Top5 pairs contribute 23/107 (21.50%), 27/107 (25.23%), and 34/107 (31.78%) conflict points.

## Findings

A. The paper's textual e_uv in {0,1} definition is closest to the binary simple graph.

B. The fixed-midpoint visual behavior with repeated parallel edges is closest to the multigraph; weighted simple edges retain multiplicity only through width.

C. Binary CI can understate repeated-conflict importance because its degree counts unique partners, not repeated conflict points.

D. Best Top10 point coverage by representation: [{'network_type': 'binary', 'l': 1, 'conflict_point_coverage': 0.5887850467289719, 'conflict_pair_coverage': 0.4444444444444444, 'covered_unique_flights': 34, 'top10_total_incident_conflict_points': 68, 'top10_mean_incident_conflict_points': 6.8, 'top10_median_incident_conflict_points': 4.5}, {'network_type': 'multi', 'l': 1, 'conflict_point_coverage': 0.48598130841121495, 'conflict_pair_coverage': 0.2916666666666667, 'covered_unique_flights': 21, 'top10_total_incident_conflict_points': 86, 'top10_mean_incident_conflict_points': 8.6, 'top10_median_incident_conflict_points': 5.0}, {'network_type': 'weighted', 'l': 1, 'conflict_point_coverage': 0.48598130841121495, 'conflict_pair_coverage': 0.2916666666666667, 'covered_unique_flights': 21, 'top10_total_incident_conflict_points': 86, 'top10_mean_incident_conflict_points': 8.6, 'top10_median_incident_conflict_points': 5.0}] (paper reference: 80.4124%, diagnostic only).

E. The metric most correlated with incident conflict-point involvement is `multi_degree` (Spearman=1.000000). Multi degree and weighted strength are expected to be highly aligned with that involvement by construction.

F. Flight 73 multi-CI ranks l1..l4: [2, 1, 1, 2]; weighted-CI ranks: [2, 1, 1, 2]. Flight 98 binary-CI ranks: [4, 1, 14, 6]; multi-CI ranks: [15, 11, 20, 6].

The evidence supports a tension between the textual binary adjacency definition and plotted network-degree behavior: **evidence suggests a discrepancy between the textual binary adjacency definition and the plotted network-degree behavior.** It does not prove the paper used a MultiGraph. Figure 8 behavior may be better explained by a multi/weighted representation because repeated edges and conflict multiplicity become visible.
