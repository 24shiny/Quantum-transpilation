import numpy as np
import re
import pennylane as qml
from penny_to_graph import Penny_to_Graph
from typing import Union, List, Union, Any, Tuple

q1 = ['Hadamard', 'PauliX', 'RY', 'U1', 'U2'] 
q2 = ['CNOT', 'CZ', 'SWAP']

# optimization methods
def extract_info_from_qnode(qnode):
    """Extracts gate info from a QNode by tracing its quantum function."""
    quantum_fn = qnode.func

    with qml.tape.QuantumTape() as tape:
        quantum_fn()

    gate_info = []
    for op in tape.operations:
        safe_params = []
        for p in op.parameters:
            try:
                safe_params.append(float(p))  
            except (TypeError, ValueError):
                safe_params.append(np.array(p).tolist()) 

        gate_info.append({
            "name": op.name,
            "wires": list(op.wires),
            "params": safe_params
        })
    
    return gate_info

def info_to_qnode(circuit_info):
    dev = qml.device('default.qubit')
    def circuit():
        for gate in circuit_info:
            if gate:
                name = gate['name']
                wires = gate['wires']
                params = gate['params']
                if name == 'Hadamard':
                    qml.Hadamard(wires=wires[0])
                elif name == 'PauliX':
                    qml.PauliX(wires=wires[0])
                elif name == 'RY':
                    qml.RY(params[0], wires=wires[0])
                elif name == 'U1':
                    qml.U1(params[0], wires=wires[0])
                elif name == 'U2':
                    params = normalize_u2_params(params)
                    if np.isclose(params, [0.0,0.0]).all():
                        pass
                    else:
                        a = extract_multiplier(params[0])
                        b = extract_multiplier(params[1])
                        qml.U2(a*np.pi, b*np.pi, wires=wires[0])
                elif name == 'CNOT':
                    qml.CNOT(wires=wires)            
                elif name == 'CZ':
                    qml.CZ(wires=wires)
                elif name == 'SWAP':
                    qml.SWAP(wires=wires)
                elif name == 'QubitUnitary':
                    matrix = np.array(params[0])
                    qml.QubitUnitary(matrix, wires=wires)
                elif name == 'U3':
                    qml.Rot(params[0], params[1], params[2], wires=wires)
                else:
                    raise ValueError(f"Unsupported gate: {name}")
        return qml.state()
    qnode = qml.QNode(circuit, dev)
    return qnode

def info_to_qnode_matrix(circuit_info): # gate merge
    with qml.tape.QuantumTape() as tape:
        for gate in circuit_info:
            name = gate['name']
            wires = gate['wires']
            params = gate['params']
            if name == 'Hadamard':
                qml.Hadamard(wires=wires[0])
            elif name == 'PauliX':
                qml.PauliX(wires=wires[0])
            elif name == 'RY':
                qml.RY(params[0], wires=wires[0])
            elif name == 'U1':
                qml.U1(params[0], wires=wires[0])
            elif name == 'U2':
                params = normalize_u2_params(params)
                if np.isclose(params, [0.0,0.0]).all():
                    pass
                else:
                    a = extract_multiplier(params[0])
                    b = extract_multiplier(params[1])
                    qml.U2(a*np.pi, b*np.pi, wires=wires[0])
            elif name == 'CNOT':
                qml.CNOT(wires=wires)            
            elif name == 'CZ':
                qml.CZ(wires=wires)
            elif name == 'SWAP':
                qml.SWAP(wires=wires)
            elif name == 'QubitUnitary':
                matrix = np.array(params[0])
                qml.QubitUnitary(matrix, wires=wires)
            else:
                raise ValueError(f"Unsupported gate: {name}")
    wires = wire_range(circuit_info)
    return name, qml.matrix(tape, wire_order=wires), wires

def info_to_qnode_matrix_lev3(circuit_info): # return name list
    names = []
    with qml.tape.QuantumTape() as tape:
        for gate in circuit_info:
            name = gate['name']
            names.append(name)
            wires = gate['wires']
            params = gate['params']
            if name == 'Hadamard':
                qml.Hadamard(wires=wires[0])
            elif name == 'PauliX':
                qml.PauliX(wires=wires[0])
            elif name == 'RY':
                qml.RY(params[0], wires=wires[0])
            elif name == 'U1':
                qml.U1(params[0], wires=wires[0])
            elif name == 'U2':
                params = normalize_u2_params(params)
                if np.isclose(params, [0.0,0.0]).all():
                    pass
                else:
                    a = extract_multiplier(params[0])
                    b = extract_multiplier(params[1])
                    qml.U2(a*np.pi, b*np.pi, wires=wires[0])
            elif name == 'CNOT':
                qml.CNOT(wires=wires)            
            elif name == 'CZ':
                qml.CZ(wires=wires)
            elif name == 'SWAP':
                qml.SWAP(wires=wires)
            elif name == 'QubitUnitary':
                matrix = np.array(params[0])
                qml.QubitUnitary(matrix, wires=wires)
            else:
                raise ValueError(f"Unsupported gate: {name}")
    wires = wire_range(circuit_info)
    return names, np.round(qml.matrix(tape, wire_order=wires),2), wires

def optimization_prep(qnode):
    pg = Penny_to_Graph(qnode)
    G = pg.G
    circuit_info = extract_info_from_qnode(qnode)
    return G, circuit_info

def community_sort(G, communities, barriers):
    def extract_index(name):
        match = re.search(r'_(\d+)$', name)
        return int(match.group(1)) if match else None

    all_communities = communities + [{barrier} for barrier in barriers]

    node_to_original = {}
    for i, community in enumerate(all_communities):
        for node in community:
            node_to_original[node] = i

    sorted_nodes = sorted(G.nodes(), key=extract_index)

    original_to_new = {}
    node_to_new = {}
    new_index = 0

    for node in sorted_nodes:
        original = node_to_original[node]
        if original not in original_to_new:
            original_to_new[original] = new_index
            new_index += 1
        node_to_new[node] = original_to_new[original]

    for node in G.nodes:
        G.nodes[node]['community'] = node_to_new[node]

    sorted_communities = [set() for _ in range(new_index)]
    for node, idx in node_to_new.items():
        sorted_communities[idx].add(node)

    return G, sorted_communities

def subcircuit_syntehsis_prep(G, communities, circuit_info):
    num_community = len(communities)
    
    subcircuit_idx_arr = []
    for i in range(num_community):
        temp_gate = [n for n in G.nodes if G.nodes[n].get('community') == i]
        temp_gate = [item for item in temp_gate if item]
        temp_com_label = [int(g.split('_')[1]) for g in temp_gate]
        subcircuit_idx_arr.append(temp_com_label)

    community_circuit_info = []
    for idx_list in subcircuit_idx_arr:
        target = [circuit_info[j] for j in idx_list]
        if target == []:
            continue
        community_circuit_info.append(target)
    return community_circuit_info

def extract_index(name):
        match = re.search(r'_(\d+)$', name)
        return int(match.group(1)) if match else None

def wire_range(gate_dic):
    wire__list = [elem['wires'] for elem in gate_dic]
    flat = [item for sublist in wire__list for item in sublist]
    if min(flat)==max(flat):
        return [min(flat)]
    else:
        return [min(flat), max(flat)]

def get_pi_multiplier(value, denominators=[1, 2, 3, 4, 6, 8, 12], tolerance=1e-9):
    if np.isclose(value, 0.0, atol=tolerance):
        return 0.0
        
    raw_ratio = value / np.pi
    abs_ratio = abs(raw_ratio)
    sign = np.sign(raw_ratio)
    
    closest_integer = np.round(abs_ratio)
    if np.isclose(abs_ratio, closest_integer, atol=tolerance):
        return sign * closest_integer

    for den in denominators:
        num_float = abs_ratio * den
        closest_num = np.round(num_float)
        
        if np.isclose(num_float, closest_num, atol=tolerance):
            return sign * closest_num / den
    return int(raw_ratio)

def extract_multiplier(angle_in_radians: Union[float, np.float64]) -> float:
    if angle_in_radians == 0.0:
        return 0.0
    multiplier = angle_in_radians / np.float64(-3.141592653589793)
    return multiplier

def normalize_u2_params(params: Union[List[Any], Tuple[Any]]) -> List[float]:
    param_array = np.array(params)
    flat_params = param_array.flatten()
    if flat_params.size < 2:
        raise ValueError(f"U2 gate requires 2 parameters, only {flat_params.size} found.")
    return flat_params[:2].tolist()