# basics
import numpy as np
import pandas as pd
import re
import networkx as nx
# penny
import pennylane as qml
from pennylane.transforms import compile
from pennylane.math import fidelity_statevector as fidelity_penny
from math import pi

def qc_com_detection(G, barriers):
    """"community detection with barriers"""
    barriers = [node for node in G.nodes if any(gate in node for gate in barriers)]
    barrier_set = set(barriers)

    G_sub = G.subgraph([node for node in G.nodes if node not in barrier_set])
    communities = list(nx.community.greedy_modularity_communities(G_sub))

    for barrier in barriers:
        communities.append({barrier})

    node_to_original_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_original_community[node] = i

    # internal function
    def extract_index(name):
        match = re.search(r'_(\d+)$', name)
        return int(match.group(1)) if match else None

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

    for node in G.nodes:
        G.nodes[node]['community'] = node_to_reindexed_community[node]
    return G, communities # updated G

def wire_range(gate_list):
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
    """takes a subcircuit as input and returns an effective unitary"""
    [w_min, w_max] = wire_range(subcircuit)

    # if w_min == w_max: # for 1q gates, return them as they are
    #     return np.array(subcircuit), np.array([w_max])

    wires = np.arange(w_min, w_max+1,1)
    num_eq = w_max - w_min + 1
    initial_matrix = np.diag(np.ones(np.power(2,num_eq)))

    # effective unitary
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

### 
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

def level_1_sythesis(effective_u_dic):
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
                    elif len(params) == 2:
                        qml.U2(params[0], params[1], wires=wires[0])
            elif dim >= 2:
                qml.QubitUnitary(params, wires=wires)
        return qml.state()
    qnode = qml.QNode(decompose_combined_u, dev)
    return qnode

def match_2q_gates(circuit_info):
    for gate in circuit_info:
        if gate['name'] == 'QubitUnitary' and gate['params']:
            matrix = np.array(gate['params'][0], dtype=complex)
            if matrix.shape == (4, 4):
                if np.allclose(matrix, CNOT_matrix):
                    gate['name'] = 'CNOT'
                elif np.allclose(matrix, CZ_matrix):
                    gate['name'] = 'CZ'
    return circuit_info # updpated 