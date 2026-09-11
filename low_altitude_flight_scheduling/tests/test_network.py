from __future__ import annotations

from src.conflict_detection import Conflict
from src.conflict_network import (
    build_conflict_network,
    collective_influence,
    network_metrics,
    select_key_flights_for_coverage,
)
from src.flight_plan import FlightPlan


def _fp(fid: int) -> FlightPlan:
    return FlightPlan(fid, (0, 0, 1), (1, 1, 1), [(0, 0, 1), (1, 1, 1)], 0.0, [0.0, 10.0], [10, 10], 0.0, 10.0)


def test_network_edges_and_ci() -> None:
    plans = [_fp(i) for i in range(5)]
    conflicts = [
        Conflict(0, 1, (0, 0, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
        Conflict(0, 2, (0, 0, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
        Conflict(0, 3, (0, 0, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
        Conflict(3, 4, (0, 0, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
    ]
    graph = build_conflict_network(plans, conflicts)
    assert graph.number_of_edges() == 4
    ci = collective_influence(graph, l=1)
    assert ci[0] >= ci[1]


def test_coverage_selection_prioritizes_repeated_conflicts() -> None:
    plans = [_fp(i) for i in range(4)]
    conflicts = [
        Conflict(0, 1, (i, 0, 1), i, i, i, i, 0, 30, "uncertain")
        for i in range(6)
    ]
    conflicts.extend(
        [
            Conflict(1, 2, (0, 1, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
            Conflict(2, 3, (0, 2, 1), 0, 0, 0, 0, 0, 30, "uncertain"),
        ]
    )
    graph = build_conflict_network(plans, conflicts)
    metrics = network_metrics(graph, ci_l=1)

    selected = select_key_flights_for_coverage(metrics, conflicts, 0.25, 4, target_coverage=0.75, max_ratio=0.5)

    assert len(selected) == 1
    assert selected[0] in {0, 1}
