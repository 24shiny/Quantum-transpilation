from qco_level_0 import *
from gate_determination import *
from collections import defaultdict
import networkx as nx
from qco_spec_table import make_spec_table
from collections import defaultdict

def optimization_level_1(qnode):
    G, circuit_info = optimization_prep(qnode)
    G, communities = graph_alg_level_1(G, barriers=['QubitUnitary'] + q2)
    community_circuit_info = subcircuit_syntehsis_level_1(G, communities, circuit_info)
    qnode_q1 = info_to_qnode(community_circuit_info)
    # df = make_spec_table(qnode, qnode_q1)
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

def split_disparate_communities(gate_list):
    grouped_gates = {}
    for gate in gate_list:
        gate_name = gate['name']
        if gate_name not in grouped_gates:
            grouped_gates[gate_name] = []
        grouped_gates[gate_name].append(gate)
    return list(grouped_gates.values())

def extract_u1(matrix):
    U = np.asarray(matrix)
    phi = np.angle(U[1, 1])
    return phi

def extract_u2(matrix):
    U = np.asarray(matrix)
    phi = np.angle(U[1, 0])
    phi_plus_lambda = np.angle(U[1, 1])
    lam = phi_plus_lambda - phi
    lam = (lam + np.pi) % (2 * np.pi) - np.pi
    return phi, lam

def extract_ry(matrix):
    U = np.asarray(matrix)
    cos_half_theta = np.real(U[0, 0])
    sin_half_theta = np.real(U[1, 0])
    half_theta = np.arctan2(sin_half_theta, cos_half_theta)
    theta = 2 * half_theta
    return theta


def subcircuit_syntehsis_level_1(G, communities, circuit_info):
    cci = subcircuit_syntehsis_prep(G, communities, circuit_info)
    community_circuit_info = []
    for ci in cci:
        community_circuit_info.extend(split_disparate_communities(ci))
    for idx, ci in enumerate(community_circuit_info):
        if len(ci) > 1:
            name, matrix, wires = info_to_qnode_matrix(ci)
            if name == 'I':
                community_circuit_info[idx] = {}         
            elif name in ['Hadamard', 'PauliX']:
                community_circuit_info[idx] =  [{'name': name, 'wires': wires, 'params':[matrix]}]
            elif name == 'U1':
                community_circuit_info[idx] =  [{'name': name, 'wires': wires, 'params':[extract_u1(matrix)]}]
            elif name == 'U2':
                community_circuit_info[idx] =  [{'name': name, 'wires': wires, 'params':[extract_u2(matrix)]}]
            elif name == 'RY':
                community_circuit_info[idx] =  [{'name': name, 'wires': wires, 'params':[extract_ry(matrix)]}]
            else:
                pass # exit

    community_circuit_info = [item for item in community_circuit_info if item]
    community_circuit_info = [gate_dict for inner_list in community_circuit_info for gate_dict in inner_list]
    return community_circuit_info