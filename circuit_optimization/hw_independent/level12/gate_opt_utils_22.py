# basics
import numpy as np
import re
import networkx as nx
from collections import defaultdict
import pennylane as qml
from penny_qiskit_utils import *
from qiskit import QuantumCircuit
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
from qiskit.transpiler import PassManager
from qiskit import QuantumCircuit
from gate_determination import * 

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
    
    # update community attribute
    attribute_name = 'community'
    has_attribute = any(attribute_name in data for _, data in G.nodes(data=True))
    if not has_attribute:
        for node in G.nodes:
            G.nodes[node].pop(attribute_name, None)

    # community index as a node attribute
    for node in G.nodes:
        G.nodes[node]['community'] = node_to_reindexed_community[node]
    return G, communities

# def wire_range(gate_list):
#     """in : gate list & out : wire range of each gate"""
#     wires = []
#     for gate in gate_list:
#         if hasattr(gate, 'wires'):
#             wires.extend(gate.wires)
#         elif hasattr(gate, 'wire'):
#             wires.append(gate.wire)
#         elif isinstance(gate, tuple) or isinstance(gate, list):
#             for item in gate:
#                 if hasattr(item, 'wires'):
#                     wires.extend(item.wires)
#                 elif hasattr(item, 'wire'):
#                     wires.append(item.wire)
#         elif hasattr(gate, '__str__'):
#             match = re.findall(r'\((\d+)\)', str(gate))
#             wires.extend([int(m) for m in match])
#     return [min(wires), max(wires)]

# def calculate_effective_u(subcircuit):
#     """in : subcircuit & out : synthesized subcircuit and its wire range"""
#     [w_min, w_max] = wire_range(subcircuit)

#     wires = np.arange(w_min, w_max+1,1)
#     num_eq = w_max - w_min + 1
#     initial_matrix = np.diag(np.ones(np.power(2,num_eq)))

#     # circuit to extract an effective unitary of a subcircuit
#     dev = qml.device('default.qubit', wires=wires)
#     @qml.qnode(dev)
#     def internal_circuit(idx):
#         qml.StatePrep(initial_matrix[idx], wires=wires)
#         for j in subcircuit:
#             qml.apply(j)
#         return qml.state()
    
#     effective_u = []
#     for idx in range(np.power(2,num_eq)):
#         effective_u.append(internal_circuit(idx))
#     effective_u = np.stack(effective_u, axis=1)
    
#     return effective_u, wires

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

def subcircuit_syntehsis_level_2(G, communities, circuit_info):
    community_circuit_info = subcircuit_syntehsis_prep(G, communities, circuit_info)

    for idx, ci in enumerate(community_circuit_info):
        if len(ci) > 1:
            matrix, wires = info_to_qnode_matrix(ci)
            # single-gate determination
            gate_name = determine_2q_gate(matrix)
            community_circuit_info[idx] = gate_info_array(gate_name, wires)

    community_circuit_info = [item[0] for item in community_circuit_info if isinstance(item, list) and item and isinstance(item[0], dict)]

    return community_circuit_info

def wire_range(gate_dic):
    wire__list = [elem['wires'] for elem in gate_dic]
    flat = [item for sublist in wire__list for item in sublist]
    if min(flat)==max(flat):
        return [min(flat)]
    else:
        return [min(flat), max(flat)]

def info_to_qnode_matrix(circuit_info):
    with qml.tape.QuantumTape() as tape:
        for gate in circuit_info:
            name = gate['name']
            wires = gate['wires']
            params = gate['params']
            if name == 'Hadamard':
                qml.Hadamard(wires=wires[0])
            elif name == 'PauliX':
                qml.PauliX(wires=wires[0])
            elif name == 'U2':
                qml.U2(params[0], params[1], wires=wires[0])
            elif name == 'CNOT':
                qml.CNOT(wires=wires)            
            elif name == 'CZ':
                qml.CZ(wires=wires)
            elif name == 'QubitUnitary':
                matrix = np.array(params[0])
                qml.QubitUnitary(matrix, wires=wires)
            else:
                raise ValueError(f"Unsupported gate: {name}")
    wires = wire_range(circuit_info)
    return qml.matrix(tape, wire_order=wires), wires

def selective_subcircuit_syntehsis(G, communities, circuit_info):
    num_community = len(communities)

    subcircuit_idx_arr = []
    for i in range(num_community):
        temp_gate = [n for n in G.nodes if G.nodes[n].get('community') == i]
        temp_gate = [item for item in temp_gate if item]
        temp_com_label = [int(g.split('_')[1]) for g in temp_gate]
        subcircuit_idx_arr.append(temp_com_label)

    # index to circuit_info
    community_circuit_info = []
    for idx_list in subcircuit_idx_arr:
        target = [circuit_info[j] for j in idx_list]
        if target == []:
            continue
        community_circuit_info.append(target)

    for idx, ci in enumerate(community_circuit_info):
        if len(ci) > 1:
            matrix, wries = info_to_qnode_matrix(ci)
            community_circuit_info[idx] =  [{'name': 'QubitUnitary', 'wires': wries, 'params':[matrix]}] # replaced

    community_circuit_info = np.array(community_circuit_info).flatten()

    return info_to_qnode(community_circuit_info)

def determine_1q_gate(params): # can be improved as the next function
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

def determine_2q_gate(params): # add defensive lines
    if is_matrix(params, CNOT_matrix):
        return 'CNOT'
    elif is_matrix(params, CZ_matrix):
        return 'CZ'
    elif is_matrix(params, CZ_CNOT_matrix):
        return ['CZ', 'CNOT']
    elif is_matrix(params, CNOT_CZ_matrix):
        return ['CNOT', 'CZ']
    elif is_matrix(params, CZ_CNOT_CZ_matrix):
        return ['CZ', 'CNOT', 'CZ']
    elif is_matrix(params, CNOT_CZ_CNOT_matrix):
        return ['CNOT', 'CZ', 'CNOT']
    
    
# def unitary_to_basis(effective_u_dic):
#     """input : dictionary of unitaries and their wire ranges & output : qnode with basis gates"""
#     num_gate = len(effective_u_dic['u'])
#     dev = qml.device('default.qubit')
#     def decompose_combined_u():
#         for i in range(num_gate):
#             params = effective_u_dic['u'][i]
#             wires = effective_u_dic['wires'][i]
#             dim = wires.shape[0]
#             if is_identity(params):
#                 continue
#             if dim == 1:
#                 if is_hadamard(params):
#                     qml.Hadamard(wires[0])
#                 elif is_x(params):
#                     qml.PauliX(wires[0])
#                 else:
#                     if len(params) == 1:
#                         qml.U1(params[0], wires=wires[0])
#                     if len(params) == 2:
#                         qml.U2(params[0], params[1], wires=wires[0])
#             elif dim == 2:
#                 if is_cnot(params):
#                     qml.CNOT(wires)
#                 elif is_cz(params):
#                     qml.CZ(wires)
#                 else:
#                     qml.QubitUnitary(params, wires=wires)
#             else:
#                 qml.QubitUnitary(params, wires=wires)
#         return qml.state()
#     qnode = qml.QNode(decompose_combined_u, dev)
#     return qnode

def unitary_to_basis_qiskit(effective_u_dic):
    # Build circuit
    error_cnt = 0
    error_gate = []
    num_gate = len(effective_u_dic['u'])
    basis = ['x', 'y', 'z', 'cx', 'cz', 'rx', 'ry', 'rz']
    qc = QuantumCircuit(max(np.concatenate(effective_u_dic['wires'])) + 1)

    for i in range(num_gate):
        params = np.round(effective_u_dic['u'][i], 3)
        wires = effective_u_dic['wires'][i].tolist()
        if is_unitary(params) and 2 ** len(wires) == params.shape[0]:
            gate = UnitaryGate(params)
            qc.unitary(gate, wires)
        else:
            error_cnt += 1
            error_gate.append(params)

    # Optimize circuit
    synth_pass = UnitarySynthesis(basis_gates=basis)
    pm = PassManager(synth_pass)
    optimized_circuit = pm.run(qc)
    return optimized_circuit

def graph_alg_level_1(G, barriers):
    """in : G, barrier list & out : updated G, community index list"""
    # set barriers and exclude them
    barriers = [node for node in G.nodes if any(gate in node for gate in barriers)]
    barrier_set = set(barriers)

    G_sub = G.subgraph([node for node in G.nodes if node not in barrier_set])
    # base_communities = list(nx.community.greedy_modularity_communities(G_sub))
    base_communities = list(nx.connected_components(G_sub.to_undirected()))

    communities = []
    for community in base_communities:
        wire_groups = defaultdict(set)
        for node in community:
            wire_set = tuple(sorted(G.nodes[node].get('wires', [])))
            wire_groups[wire_set].add(node)
        communities.extend(wire_groups.values())
    return community_sort(G, communities, barriers)

def graph_alg_level_2(G, barriers):
    """in : G, barrier list & out : updated G, community index list"""
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

### level_3
# def get_subgraph(G):
#     subgraphs = []
#     gate_2q = [n for n, attr in G.nodes(data=True) if attr['num_q'] == 2]

#     for center in gate_2q:
#         radius = 0
#         wires = G.nodes[center]['wires']
#         prev_subG = None

#         while True:
#             bool_list = []
#             subG = nx.ego_graph(G.to_undirected(), center, radius=radius)
#             for node, attr in subG.nodes(data=True):
#                 bool_list.append(set(attr['wires']).issubset(wires))        

#             if bool_list.count(False) > 1:
#                 if prev_subG is not None:
#                     subgraphs.append({'center': center, 'wires':wires, 'subG': prev_subG})
#                 break
#             else:
#                 prev_subG = subG 
#                 radius += 1
#     return subgraphs

# revised on Oct 22th
def between_nodes(G_directed, source, target):
    G = nx.DiGraph(G_directed)
    paths = list(nx.all_simple_paths(G, source=source, target=target))
    flat_nodes = set()
    for path in paths:
        flat_nodes.update(path)
    flat_nodes.remove(source)
    flat_nodes.remove(target)
    return flat_nodes

def get_wire(G, node):
    return G.nodes[node]['wires']

def get_subgraph(G): 
    subgraphs = []
    G_directed = G
    G_undirected = G_directed.to_undirected()
    gate_2q = [n for n, attr in G_undirected.nodes(data=True) if attr.get('num_q') == 2]

    for center in gate_2q:
        reference_wires = set(get_wire(G_undirected, center))
        visited = set()
        stack = [center]
        valid_nodes = set()

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            current_wires = set(get_wire(G_undirected, current))
            if not current_wires.issubset(reference_wires):
                continue

            valid_nodes.add(current)

            # Predecessors
            for pred in G_directed.predecessors(current):
                if pred in visited:
                    continue

                pred_wires = set(get_wire(G_undirected, pred))
                if not pred_wires.issubset(reference_wires):
                    continue

                if G_directed.nodes[pred].get('num_q') == 2:
                    succ = between_nodes(G_directed, pred, current)
                    if len(succ) == 0:
                        continue
                    succ_wires = []
                    for s in succ:
                        succ_wires.extend(get_wire(G_undirected, s))
                    if not set(succ_wires).issubset(reference_wires):
                        continue

                stack.append(pred)

            # Successors
            for succ in G_directed.successors(current):
                if succ in visited:
                    continue

                succ_wires = set(get_wire(G_undirected, succ))
                if not succ_wires.issubset(reference_wires):
                    continue

                if G_directed.nodes[succ].get('num_q') == 2:
                    preds = between_nodes(G_directed, current, succ)
                    if len(preds) == 0:
                        continue
                    preds_wires = []
                    for p in preds:
                        preds_wires.extend(get_wire(G_undirected, p))
                    if not set(preds_wires).issubset(reference_wires):
                        continue

                stack.append(succ)

        if len(valid_nodes) > 1:
            subG = G_undirected.subgraph(valid_nodes).copy()
            subgraphs.append({'center': center, 'wires': list(reference_wires), 'subG': subG})
    return subgraphs

# def subgraph_trimming(subgraphs):
#     new_subgraph = [elem for elem in subgraphs if len(elem['subG'].nodes()) > 1]
#     for elem in new_subgraph:
#         reference_wires = set(elem['wires'])
#         subG = elem['subG']
#         nodes_to_remove = [node for node, attr in subG.nodes(data=True) if not set(attr.get('wires', [])).issubset(reference_wires)]
#         subG.remove_nodes_from(nodes_to_remove)
#     return new_subgraph

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
    # for n, attr in G.nodes(data=True):
    #     G.nodes[n]['num_q'] = len(attr.get('wires', []))

    subgraphs = get_subgraph(G)
    # new_subgraph = subgraph_trimming(subgraphs)
    unique_subgraphs = get_unique_subgraphs(subgraphs)
    communities = []
    for elem in unique_subgraphs:
        communities.append(set(elem['nodes']))
    return communities

def graph_alg_level_3(G):
    communities = get_communities(G)
    barriers = [n for n in list(G.nodes()) if n not in set().union(*communities)]
    # print(len(G.nodes), len(barriers))
    return community_sort(G, communities, barriers)
    