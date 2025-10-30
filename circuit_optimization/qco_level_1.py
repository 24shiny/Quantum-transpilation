from qco_level_0 import *
from gate_determination import *
from collections import defaultdict
import networkx as nx
from qco_spec_table import make_spec_table, show_circuit

def optimization_level_1(qnode):
    G, circuit_info = optimization_prep(qnode)
    G, communities = graph_alg_level_1(G, barriers=['QubitUnitary'] + q2)
    community_circuit_info = subcircuit_syntehsis_level_1(G, communities, circuit_info)
    qnode_q1 = info_to_qnode(community_circuit_info)
    
    print(make_spec_table(qnode, qnode_q1))
    show_circuit(qnode_q1)

    return qnode_q1

def graph_alg_level_1(G, barriers):
    barriers = [node for node in G.nodes if any(gate in node for gate in barriers)]
    barrier_set = set(barriers)

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

def subcircuit_syntehsis_level_1(G, communities, circuit_info):
    community_circuit_info = subcircuit_syntehsis_prep(G, communities, circuit_info)

    for idx, ci in enumerate(community_circuit_info):
        if len(ci) > 1:
            matrix, wires = info_to_qnode_matrix(ci)
            # single-gate determination
            gate_name = determine_1q_gate(matrix)
            if gate_name == 'I':
                community_circuit_info[idx] = {} # clear the gate information
            else:
                community_circuit_info[idx] =  [{'name': gate_name, 'wires': wires, 'params':[matrix]}] # replaced

    community_circuit_info = [item[0] for item in community_circuit_info if isinstance(item, list) and item and isinstance(item[0], dict)]

    return community_circuit_info

def determine_1q_gate(params):
    if is_identity(params):
        return 'I'
    elif is_matrix(params, H_matrix):
        return 'Hadamard'
    elif is_matrix(params, X_matrix):
        return 'PauliX'
    else:
        if len(params) == 1:
            return 'U1'
        elif len(params) == 2:
            return 'U2'