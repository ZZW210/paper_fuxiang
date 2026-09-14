import networkx as nx

from run_multiline_baseline_full import multi_ci_l2


def test_multi_ci_uses_multiplicity_but_simple_topology_shells():
    topology = nx.Graph()
    topology.add_edges_from([(0, 1), (1, 2)])
    multi = nx.MultiGraph()
    multi.add_nodes_from(topology.nodes)
    multi.add_edges_from([(0, 1)] * 5 + [(1, 2)] * 3)
    scores = multi_ci_l2(topology, multi)
    assert dict(multi.degree()) == {0: 5, 1: 8, 2: 3}
    assert scores[0] == (5 - 1) * (3 - 1)
    assert scores[1] == 0
