import networkx as nx

from analyze_conflict_network_representation import shell_ci


def test_multi_degree_ci_keeps_simple_shell():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2)])
    assert shell_ci(graph, {0: 3, 1: 4, 2: 1}, 2)[0] == 0
