from qco_level_0 import *
from gate_determination import *
from collections import defaultdict
import networkx as nx
from qco_spec_table import make_spec_table

def optimization_level_2(qnode):
    G, circuit_info = optimization_prep(qnode)
    G, communities = graph_alg_level_2(G, barriers=['QubitUnitary'] + q1)
    community_circuit_info = subcircuit_syntehsis_level_2(G, communities, circuit_info)
    qnode_q2 = info_to_qnode(community_circuit_info)
    # print(make_spec_table(qnode, qnode_q2))
    return qnode_q2
    
def graph_alg_level_2(G, barriers):
    barriers = [node for node in G.nodes if any(gate in node for gate in barriers)]
    barrier_set = set(barriers)
    
    gate_2q = [n for n, attr in G.nodes(data=True) if attr.get('num_q') == 2]
    edge_to_remove = []
    for e in G.edges():
        if (e[0] in gate_2q) and (e[1] in gate_2q):
            if G.number_of_edges(e[0], e[1]) != 2:
                edge_to_remove.append(e)           
    G.remove_edges_from(set(edge_to_remove))

    G_sub = G.subgraph([node for node in G.nodes if node not in barrier_set])
    base_communities = list(nx.connected_components(G_sub.to_undirected()))

    communities = []
    for community in base_communities:
        wire_groups = defaultdict(set)
        for node in community:
            wire_set = tuple(sorted(G.nodes[node].get('wires', [])))
            wire_groups[wire_set].add(node)
        communities.extend(wire_groups.values())
    
    G, communities = community_sort(G, communities, barriers)
    communities = [sorted(set(gate_set), key=extract_index) for gate_set in communities]   
    return G, communities

def subcircuit_syntehsis_level_2(G, communities, circuit_info):
    community_circuit_info = subcircuit_syntehsis_prep(G, communities, circuit_info)

    for idx, ci in enumerate(community_circuit_info):
        if len(ci) > 1:
            name, matrix, wires = info_to_qnode_matrix(ci)
            gate_name = determine_2q_gate(matrix)
            if gate_name:
                community_circuit_info[idx] = gate_info_array(gate_name, wires)
            else:
                pass # keep as they are

    community_circuit_info = [gate_dict for inner_list in community_circuit_info for gate_dict in inner_list]
    return community_circuit_info

def determine_2q_gate(params):
    U = np.asarray(params, dtype=complex)
    if is_identity(params):
        return ['I']
    for name, ref_matrix in two_q_gates_extended.items():
        if np.allclose(U, ref_matrix):
            return name.split('_')
    return False # undetermined
    
def gate_info_array(gates, wires):
    temp = []
    for g in gates:
        if g == 'I':
            continue
        elif g == 'CNOT':
            temp.append({'name': 'CNOT', 'wires': wires, 'params':[two_q_gates_extended['CNOT']]})
        elif g == 'CNOTinv':
            temp.append({'name': 'CNOT', 'wires': wires[::-1], 'params':[two_q_gates_extended['CNOTinv']]})            
        elif g == 'CZ':
            temp.append({'name': 'CZ', 'wires': wires, 'params':[two_q_gates_extended['CZ']]})
        elif g == 'SWAP':
            temp.append({'name': 'SWAP', 'wires': wires, 'params':[two_q_gates_extended['SWAP']]})
    return temp 