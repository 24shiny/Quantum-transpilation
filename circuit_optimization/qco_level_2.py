from qco_level_0 import *
from gate_determination import *
from collections import defaultdict
import networkx as nx

def optimization_level_2(qnode):
    G, circuit_info = optimization_prep(qnode)

    # community detection excluding barriers
    G, communities = graph_alg_level_2(G, barriers=['QubitUnitary'] + q1)
    community_circuit_info = subcircuit_syntehsis_level_2(G, communities, circuit_info)
    qnode_q2 = info_to_qnode(community_circuit_info)

    # print results
    print(summary_penny(qnode_q2))
    qml.draw_mpl(qnode_q2)()
    plt.show()

    return qnode_q2

def graph_alg_level_2(G, barriers):
    # set barriers and exclude them
    barriers = [node for node in G.nodes if any(gate in node for gate in barriers)]
    barrier_set = set(barriers)

    def is_barrier(barrier_set, shared_nodes):
        check = all(item in barrier_set for item in shared_nodes)
        return check
    
    # remove shared 1-q neighbors
    edge_to_remove = []
    for e in G.edges():
        gate_2q = [n for n, attr in G.nodes(data=True) if attr.get('num_q') == 2]
        gate_1 = e[0]
        gate_2 = e[1]
        if gate_1 in gate_2q and gate_2 in gate_2q:
            neighbors1 = set(G.to_undirected().neighbors(gate_1))
            neighbors2 = set(G.to_undirected().neighbors(gate_2))
            shared_nodes = neighbors1.intersection(neighbors2)
            if shared_nodes and is_barrier(barrier_set, shared_nodes):
                edge_to_remove.append(e)
    G.remove_edges_from(edge_to_remove)

    G_sub = G.subgraph([node for node in G.nodes if node not in barrier_set])
    base_communities = list(nx.connected_components(G_sub.to_undirected()))

    communities = []
    for community in base_communities:
        wire_groups = defaultdict(set)
        for node in community:
            wire_set = tuple(sorted(G.nodes[node].get('wires', [])))
            wire_groups[wire_set].add(node)
        communities.extend(wire_groups.values())
    return community_sort(G, communities, barriers)

def subcircuit_syntehsis_level_2(G, communities, circuit_info):
    community_circuit_info = subcircuit_syntehsis_prep(G, communities, circuit_info)

    for idx, ci in enumerate(community_circuit_info):
        if len(ci) > 1:
            matrix, wires = info_to_qnode_matrix(ci)
            # single-gate determination
            gate_name = determine_2q_gate(matrix)
            community_circuit_info[idx] = gate_info_array(gate_name, wires)

    community_circuit_info = [item[0] for item in community_circuit_info if isinstance(item, list) and item and isinstance(item[0], dict)]
    community_circuit_info = [gate for gate in community_circuit_info if gate]

    return community_circuit_info

def determine_2q_gate(params): # add defensive lines
    if is_identity(params):
        return ['I']
    elif is_matrix(params, CNOT_matrix):
        return ['CNOT']
    elif is_matrix(params, CZ_matrix):
        return ['CZ']
    elif is_matrix(params, CZ_CNOT_matrix):
        return ['CZ', 'CNOT']
    elif is_matrix(params, CNOT_CZ_matrix):
        return ['CNOT', 'CZ']
    elif is_matrix(params, CZ_CNOT_CZ_matrix):
        return ['CZ', 'CNOT', 'CZ']
    elif is_matrix(params, CNOT_CZ_CNOT_matrix):
        return ['CNOT', 'CZ', 'CNOT']
    else:
        return False
    
def gate_info_array(gates, wires):
    temp = []
    for g in gates:
        if g == 'I':
            temp.append({})
        elif g == 'CNOT':
            temp.append({'name': g, 'wires': wires, 'params':[CNOT_matrix]})
        elif g == 'CZ':
            temp.append({'name': g, 'wires': wires, 'params':[CZ_matrix]})
    return temp 