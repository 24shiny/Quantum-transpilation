# basics
import numpy as np
import re
import networkx as nx
from collections import defaultdict
import pennylane as qml
from penny_qiskit_utils import *

def community_sort(G, communities, barriers):
    # barriers as singleton communities
    for barrier in barriers:
        communities.append({barrier})

    # rearange community indices 
    node_to_original_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_original_community[node] = i

    ##### internal function
    def extract_index(name):
        match = re.search(r'_(\d+)$', name)
        return int(match.group(1)) if match else None
    #####

    sorted_nodes = sorted(G.nodes(), key=extract_index)

    original_to_new_index = {}
    new_index = 0
    node_to_reindexed_community = {}

    for node in sorted_nodes:
        original = node_to_original_community[node]
        if original not in original_to_new_index:
            original_to_new_index[original] = new_index
            new_index += 1
        node_to_reindexed_community[node] = original_to_new_index[original]

    # community index as a node attribute
    for node in G.nodes:
        G.nodes[node]['community'] = node_to_reindexed_community[node]
    return G, communities

def graph_alg_level_1(G, barriers):
    """in : G, barrier list & out : updated G, community index list"""
    # set barriers and exclude them
    barriers = [node for node in G.nodes if any(gate in node for gate in barriers)]
    barrier_set = set(barriers)

    G_sub = G.subgraph([node for node in G.nodes if node not in barrier_set])
    base_communities = list(nx.community.greedy_modularity_communities(G_sub))

    communities = []
    for community in base_communities:
        wire_groups = defaultdict(set)
        for node in community:
            wire_set = tuple(sorted(G.nodes[node].get('wires', [])))
            wire_groups[wire_set].add(node)
        communities.extend(wire_groups.values())
    return community_sort(G, communities, barriers)

def graph_alg_level_3(G, communities):
    barriers = [n for n in list(G.nodes()) if n not in set().union(*communities)]
    print(len(G.nodes), len(barriers))
    return community_sort(G, communities, barriers)

def wire_range(gate_list):
    """in : gate list & out : wire range of each gate"""
    wires = []
    for gate in gate_list:
        if hasattr(gate, 'wires'):
            wires.extend(gate.wires)
        elif hasattr(gate, 'wire'):
            wires.append(gate.wire)
        elif isinstance(gate, tuple) or isinstance(gate, list):
            for item in gate:
                if hasattr(item, 'wires'):
                    wires.extend(item.wires)
                elif hasattr(item, 'wire'):
                    wires.append(item.wire)
        elif hasattr(gate, '__str__'):
            match = re.findall(r'\((\d+)\)', str(gate))
            wires.extend([int(m) for m in match])
    return [min(wires), max(wires)]

def calculate_effective_u(subcircuit):
    """in : subcircuit & out : synthesized subcircuit and its wire range"""
    [w_min, w_max] = wire_range(subcircuit)

    wires = np.arange(w_min, w_max+1,1)
    num_eq = w_max - w_min + 1
    initial_matrix = np.diag(np.ones(np.power(2,num_eq)))

    # circuit to extract an effective unitary of a subcircuit
    dev = qml.device('default.qubit', wires=wires)
    @qml.qnode(dev)
    def internal_circuit(idx):
        qml.StatePrep(initial_matrix[idx], wires=wires)
        for j in subcircuit:
            qml.apply(j)
        return qml.state()
    
    effective_u = []
    for idx in range(np.power(2,num_eq)):
        effective_u.append(internal_circuit(idx))
    effective_u = np.stack(effective_u, axis=1)
    
    return effective_u, wires

def subcircuit_syntehsis(G, ds, communities, seed=0):
    """In : G, updated ds, community index list and seed 
    & Out : dictionary of effective unitaries and their wire ranges"""
    num_community = len(communities)
    # idx
    subcircuit_idx_arr = []
    for i in range(num_community):
        temp_gate = [n for n in G.nodes if G.nodes[n].get('community') == i]
        temp_com_label = [int(g.split('_')[1]) for g in temp_gate]
        subcircuit_idx_arr.append(temp_com_label)

    # gate list
    subcircuit_gate_arr = []
    for i in range(num_community):
        subcircuit_gate_arr.append([ds.circuits[seed][j] for j in subcircuit_idx_arr[i]])

    u_arr = []
    wire_arr = []
    for i in range(num_community):
        u_temp, w_temp = calculate_effective_u(subcircuit_gate_arr[i])
        u_arr.append(u_temp)
        wire_arr.append(w_temp)

    effective_u_dic = {}
    effective_u_dic['u'] = u_arr
    effective_u_dic['wires'] = wire_arr

    return effective_u_dic

###  methods for unitary_to_basis()
def is_identity(gate):
    return np.allclose(gate, np.eye(len(gate)))

def is_hadamard(gate):
    h = np.array([[ 0.70710678+0.j,  0.70710678+0.j],
       [ 0.70710678+0.j, -0.70710678+0.j]])
    return np.allclose(gate, h)

def is_x(gate):
    x = np.array([[0,1], [1,0]], dtype=complex)
    return np.allclose(gate, x)

CNOT_matrix = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
], dtype=complex)

def is_cnot(gate):
    return np.allclose(gate, CNOT_matrix)

CZ_matrix = np.array([
[1, 0, 0, 0],
[0, 1, 0, 0],
[0, 0, 1, 0],
[0, 0, 0, -1]
], dtype=complex)

def is_cz(gate):
    return np.allclose(gate, CZ_matrix)
###

def unitary_to_basis(effective_u_dic):
    """input : dictionary of unitaries and their wire ranges & output : qnode with basis gates"""
    num_gate = len(effective_u_dic['u'])
    dev = qml.device('default.qubit')
    def decompose_combined_u():
        for i in range(num_gate):
            params = effective_u_dic['u'][i]
            wires = effective_u_dic['wires'][i]
            dim = wires.shape[0]
            if is_identity(params):
                continue
            if dim == 1:
                if is_hadamard(params):
                    qml.Hadamard(wires[0])
                elif is_x(params):
                    qml.PauliX(wires[0])
                else:
                    if len(params) == 1:
                        qml.U1(params[0], wires=wires[0])
                    if len(params) == 2:
                        qml.U2(params[0], params[1], wires=wires[0])
            elif dim == 2:
                if is_cnot(params):
                    qml.CNOT(wires)
                elif is_cz(params):
                    qml.CZ(wires)
                else:
                    qml.QubitUnitary(params, wires=wires)
            else:
                qml.QubitUnitary(params, wires=wires)
        return qml.state()
    qnode = qml.QNode(decompose_combined_u, dev)
    return qnode

def graph_alg_level_2(G, barriers):
    """in : G, barrier list & out : updated G, community index list"""
    # set barriers and exclude them
    barriers = [node for node in G.nodes if any(gate in node for gate in barriers)]
    G.remove_nodes_from(barriers)

    edge_counts = defaultdict(int)
    for u, v, _ in G.edges(keys=True):
        edge_counts[(u, v)] += 1

    multi_edge_pairs = [pair for pair, count in edge_counts.items() if count >= 2]
    node_to_remove = np.array(multi_edge_pairs).flatten()
    idx_to_remove = [int(i.split('_')[1]) for i in node_to_remove]

    return sorted(idx_to_remove, reverse=True)

### level_3
def get_subgraph(G):
    subgraphs = []
    gate_2q = [n for n, attr in G.nodes(data=True) if attr['num_q'] == 2]

    for center in gate_2q:
        radius = 0
        wires = G.nodes[center]['wires']
        prev_subG = None

        while True:
            bool_list = []
            subG = nx.ego_graph(G.to_undirected(), center, radius=radius)
            for node, attr in subG.nodes(data=True):
                bool_list.append(set(attr['wires']).issubset(wires))        

            if bool_list.count(False) > 1:
                if prev_subG is not None:
                    subgraphs.append({'center': center, 'wires':wires, 'subG': prev_subG})
                break
            else:
                prev_subG = subG 
                radius += 1
    return subgraphs

def subgraph_trimming(subgraphs):
    new_subgraph = [elem for elem in subgraphs if len(elem['subG'].nodes()) > 1]
    for elem in new_subgraph:
        reference_wires = set(elem['wires'])
        subG = elem['subG']
        nodes_to_remove = [node for node, attr in subG.nodes(data=True) if not set(attr.get('wires', [])).issubset(reference_wires)]
        subG.remove_nodes_from(nodes_to_remove)
    return new_subgraph

def get_unique_subgraphs(new_subgraph):
    # compare two adjacent graphs
    unique_subgraphs = [new_subgraph[0]]

    for i in range(1, len(new_subgraph)):
        prev_nodes = set(new_subgraph[i - 1]['subG'].nodes())
        curr_nodes = set(new_subgraph[i]['subG'].nodes())

        # If they share nodes, keep the one with more nodes
        if prev_nodes & curr_nodes:
            if len(curr_nodes) > len(prev_nodes):
                unique_subgraphs[-1] = new_subgraph[i] 
        else:
            unique_subgraphs.append(new_subgraph[i])
            
    for idx, elem in enumerate(unique_subgraphs):
     unique_subgraphs[idx]['nodes'] = list(elem['subG'].nodes())
    return unique_subgraphs

def get_communities(G):
    # final toutch - number of qubits 
    for n, attr in G.nodes(data=True):
        G.nodes[n]['num_q'] = len(attr.get('wires', []))

    subgraphs = get_subgraph(G)
    new_subgraph = subgraph_trimming(subgraphs)
    unique_subgraphs = get_unique_subgraphs(new_subgraph)
    communities = []
    for elem in unique_subgraphs:
        communities.append(set(elem['nodes']))
    return communities
    