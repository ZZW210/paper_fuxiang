import networkx as nx

from diagnose_multiline_ci_radius import radius_scores


def test_radius_scores_use_simple_shell_and_multi_degree():
    topology = nx.Graph([(0, 1), (1, 2)])
    multi = nx.MultiGraph()
    multi.add_nodes_from(topology.nodes)
    multi.add_edges_from([(0, 1)] * 5 + [(1, 2)] * 3)
    degrees, scores, shells = radius_scores(topology, multi, (1, 2))
    assert degrees == {0: 5, 1: 8, 2: 3}
    assert shells[1][0] == [1]
    assert shells[2][0] == [2]
    assert scores[1][0] == (5 - 1) * (8 - 1)
    assert scores[2][0] == (5 - 1) * (3 - 1)
