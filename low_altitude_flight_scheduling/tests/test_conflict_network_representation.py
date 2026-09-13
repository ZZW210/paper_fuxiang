import networkx as nx

from analyze_conflict_network_representation import shell_ci


def test_multigraph_ci_uses_parallel_degree_and_simple_topology_shells():
    topology = nx.Graph(); topology.add_edges_from([(0, 1), (1, 2)])
    multi_degree = {0: 3, 1: 4, 2: 1}
    assert shell_ci(topology, multi_degree, 1)[0] == 6
    assert shell_ci(topology, multi_degree, 2)[0] == 0
