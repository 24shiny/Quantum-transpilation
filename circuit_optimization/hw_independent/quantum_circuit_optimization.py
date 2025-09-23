# basics
import numpy as np
import matplotlib.pyplot as plt
# penny
import pennylane as qml
from pennylane.transforms import *
# self-developed
from gate_opt_utils import *
from penny_qiskit_utils import *
from penny_to_graph_29 import Penny_to_Graph

###### main

q1 = ['Hadamard', 'PauliX', 'RY', 'U1', 'U2']
q2 = ['CNOT', 'CZ']

def remove_barrier(qnode):
    circuit_info = extract_info_from_qnode(qnode)
    
    filtered = []
    count = 0 
    for gate in circuit_info:
        if gate['params']:
            if isinstance(gate['params'][0], list):
                matrix = np.array(gate['params'][0])
                if is_identity(matrix):
                    count += 1
                    continue  # Skip identity matrix
        filtered.append(gate)
    print(f'{count} barriers are removed')
    return info_to_qnode(filtered)

def optimization_lev_1(ds, qnode, seed=0):
    """optimization w.r.t 1-qubit gates"""
    circuit_info_q1 = extract_info_from_qnode(qnode) # same with circuit_info
    for idx, dic in enumerate(circuit_info_q1):
        if dic['name'] in q2:
            circuit_info_q1[idx] = {'name': 'QubitUnitary', 'wires': dic['wires'], 'params': [np.eye(4)]}
    qnode_q1 = info_to_qnode(circuit_info_q1)

    # to graph
    pg = Penny_to_Graph(qnode_q1)
    G = pg.G

    # community detection excluding barriers
    G, communities = graph_alg_level_1(G, barriers=['QubitUnitary'])
    effective_u_dic = subcircuit_syntehsis(G, ds, communities, seed)
    qnode_q1 = unitary_to_basis(effective_u_dic)

    # print results
    print(summary_penny(qnode_q1))
    qml.draw_mpl(qnode_q1)()
    plt.show()

    return qnode_q1 # returns an optimized qnode

def optimization_lev_2(ds, qnode, seed=0):
    """optimization w.r.t 2-qubit gates"""
    circuit_info = extract_info_from_qnode(qnode)
    circuit_info_q2 = circuit_info.copy()

    for idx, dic in enumerate(circuit_info_q2):
        if dic['name'] in q1:
            circuit_info_q2[idx] = {'name': 'QubitUnitary', 'wires': dic['wires'], 'params': [np.eye(2)]}
    qnode_q2 = info_to_qnode(circuit_info_q2)
    
    # to graph
    pg = Penny_to_Graph(qnode_q2)
    G = pg.G

    # 2-qubit gate cancellation
    idx_to_remove = graph_alg_level_2(G, barriers=['QubitUnitary'])
    for i in idx_to_remove:
        del circuit_info[i]
        del ds.circuits[seed][i]

    qnode_q2 = info_to_qnode(circuit_info)

    return ds, qnode_q2



