from qco_level_0 import *
from gate_determination import *
import networkx as nx
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
from qiskit.transpiler import PassManager
from qiskit import QuantumCircuit
from qco_spec_table import make_spec_table

def optimization_level_3(qnode):
    G, circuit_info = optimization_prep(qnode)  
    G, communities = graph_alg_level_3(G, barriers=['QubitUnitary']) # working on
    community_circuit_info = subcircuit_syntehsis_level_3(G, communities, circuit_info)
    qnode_q3 = info_to_qnode(community_circuit_info)
    df = make_spec_table(qnode, qnode_q3)
    return qnode_q3, df

def graph_alg_level_3(G, barriers):
    barriers = [node for node in G.nodes if any(gate in node for gate in barriers)]
    barrier_set = set(barriers)

    G_sub = G.subgraph([node for node in G.nodes if node not in barrier_set])
    communities = get_communities(G_sub)

    barriers = [n for n in list(G.nodes()) if n not in set().union(*communities)]
    for barrier in barriers:
        communities.append({barrier})

    G, communities = community_sort(G, communities, barriers)
    community_topological_sort(G, communities)
    communities = [sorted(set(gate_set), key=extract_index) for gate_set in communities]
    return G, communities

# def subcircuit_syntehsis_level_3_qiskit(G, communities, circuit_info):
#     community_circuit_info = subcircuit_syntehsis_prep(G, communities, circuit_info)

#     for idx, ci in enumerate(community_circuit_info):
#         if len(ci) > 1:
#             matrix, wires = info_to_qnode_matrix(ci)
#             community_circuit_info[idx] = wire_mapping(qiskit_optimization(matrix), wires)

#     community_circuit_info = [item[0] for item in community_circuit_info if isinstance(item, list) and item and isinstance(item[0], dict)]
#     community_circuit_info = [gate for gate in community_circuit_info if gate]

#     return info_to_qnode(community_circuit_info)

def find_multiple_sequences(main_list):
    found_matches = []
    len_main = len(main_list)
    targets = [['Hadamard', 'CZ', 'Hadamard'],['Hadamard', 'CNOT', 'Hadamard'],['PauliX', 'CNOT', 'PauliX']]
    
    for sub_sequence in targets:
        len_sub = len(sub_sequence)
        if len_sub > len_main:
            continue 
        for i in range(len_main - len_sub + 1):
            window = main_list[i : i + len_sub]
            if window == sub_sequence:
                found_matches.append(sub_sequence)
                break
    if len(found_matches) == 0:
        return False
    else:
        return True
    
def subcircuit_syntehsis_level_3(G, communities, circuit_info):
    community_circuit_info = subcircuit_syntehsis_prep(G, communities, circuit_info)

    for idx, ci in enumerate(community_circuit_info):
        if len(ci) > 2:
            names, matrix, wires = info_to_qnode_matrix_lev3(ci)
            if find_multiple_sequences(names):
                gate_name = determine_12q_gate(matrix)
                if gate_name:
                    community_circuit_info[idx] = gate_info_array(gate_name, wires)
                else:
                    pass # keep as they are

    community_circuit_info = [gate_dict for inner_list in community_circuit_info for gate_dict in inner_list]
    return community_circuit_info

def determine_12q_gate(params):
    U = np.asarray(params, dtype=complex)
    if is_identity(params):
        return ['I']
    for name, ref_matrix in two_q_gates.items():
        if np.allclose(U, ref_matrix):
            return name.split('_')
    return False # undetermined

def gate_info_array(gates, wires):
    temp = []
    for g in gates:
        if g == 'I':
            continue
        elif g == 'PauliX':
            temp.append({'name': 'PauliX', 'wires': [wires[0]], 'params':[]})
        elif g == 'Hadamard':
            temp.append({'name': 'Hadamard', 'wires': [wires[0]], 'params':[]})
        elif g == 'CNOT':
            temp.append({'name': 'CNOT', 'wires': wires, 'params':[]})
        elif g == 'CNOTinv':
            temp.append({'name': 'CNOT', 'wires': wires[::-1], 'params':[]})            
        elif g == 'CZ':
            temp.append({'name': 'CZ', 'wires': wires, 'params':[]})
        elif g == 'SWAP':
            temp.append({'name': 'SWAP', 'wires': wires, 'params':[]})
    return temp 

# def get_unique_subgraphs(new_subgraph):
#     # compare two adjacent graphs
#     unique_subgraphs = [new_subgraph[0]]

#     for i in range(1, len(new_subgraph)):
#         prev_nodes = set(new_subgraph[i - 1]['subG'].nodes())
#         curr_nodes = set(new_subgraph[i]['subG'].nodes())

#         # If they share nodes, keep the one with more nodes
#         if prev_nodes & curr_nodes:
#             if len(curr_nodes) > len(prev_nodes):
#                 unique_subgraphs[-1] = new_subgraph[i] 
#         else:
#             unique_subgraphs.append(new_subgraph[i])
            
#     for idx, elem in enumerate(unique_subgraphs):
#      unique_subgraphs[idx]['nodes'] = list(elem['subG'].nodes())
#     return unique_subgraphs

# def get_communities(G_sub):
#     subgraphs = get_subgraph(G_sub)
#     unique_subgraphs = get_unique_subgraphs(subgraphs)

#     communities = []
#     for ug in unique_subgraphs:
#         communities.append(list(ug.nodes()))
#     return communities

def community_topological_sort(G, communities):
    node_to_community = {}
    for idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = idx

    # mata graph
    meta_graph = nx.DiGraph()
    meta_graph.add_nodes_from(range(len(communities)))

    for u, v in G.edges():
        cu = node_to_community.get(u)
        cv = node_to_community.get(v)
        if cu is not None and cv is not None and cu != cv:
            meta_graph.add_edge(cu, cv)

    try: # topological sort
        sorted_indices = list(nx.topological_sort(meta_graph))
    except nx.NetworkXUnfeasible:
        raise ValueError("Community dependencies contain cycles — cannot sort.")

    # reindexing
    sorted_communities = [set() for _ in sorted_indices]
    for new_idx, old_idx in enumerate(sorted_indices):
        for node in communities[old_idx]:
            G.nodes[node]['community'] = new_idx
            sorted_communities[new_idx].add(node)

    return G, sorted_communities

# def wire_mapping(circuit_info, wires):
#     wire_map = {0: wires[0], 1: wires[1]}
#     remapped_gates = []
#     for gate in circuit_info:
#         remapped_gate = gate.copy()
#         remapped_gate['wires'] = [wire_map[w] for w in gate['wires']]
#         remapped_gates.append(remapped_gate)
#     return remapped_gates

# def qiskit_optimization(u):
#     basis = ['h', 'x', 'cx', 'cz', 'ry', 'u', 'u1', 'u2'] 
#     qc = QuantumCircuit(2)
#     qc.unitary(u, range(2))
#     synth_pass = UnitarySynthesis(basis_gates=basis) # basis_gates=basis
#     pm = PassManager(synth_pass)
#     optimized_circuit = pm.run(qc)
#     qnoe_temp = qml.from_qiskit(optimized_circuit)
#     dev = qml.device("default.qubit", wires=2)
#     return extract_info_from_qnode(qml.QNode(qnoe_temp, dev))

def get_wire(G, node):
    return G.nodes[node]['wires']

def between_nodes(G_directed, source, target):
    G = nx.DiGraph(G_directed)
    paths = list(nx.all_simple_paths(G, source=source, target=target))
    flat_nodes = set()
    for path in paths:
        flat_nodes.update(path)
    flat_nodes.remove(source)
    flat_nodes.remove(target)
    return flat_nodes

# def get_subgraph(G_sub): 
#     subgraphs = []
#     G_directed = G_sub
#     G_undirected = G_directed.to_undirected()
#     gate_2q = [n for n, attr in G_undirected.nodes(data=True) if attr.get('num_q') == 2]

#     for center in gate_2q:
#         reference_wires = set(get_wire(G_undirected, center))
#         visited = set()
#         stack = [center]
#         valid_nodes = set()

#         while stack:
#             current = stack.pop()
#             if current in visited:
#                 continue
#             visited.add(current)

#             current_wires = set(get_wire(G_undirected, current))
#             if not current_wires.issubset(reference_wires):
#                 continue

#             valid_nodes.add(current)

#             # Predecessors
#             for pred in G_directed.predecessors(current):
#                 if pred in visited:
#                     continue

#                 pred_wires = set(get_wire(G_undirected, pred))
#                 if not pred_wires.issubset(reference_wires):
#                     continue

#                 if G_directed.nodes[pred].get('num_q') == 2:
#                     succ = between_nodes(G_directed, pred, current)
#                     if len(succ) == 0:
#                         continue
#                     succ_wires = []
#                     for s in succ:
#                         succ_wires.extend(get_wire(G_undirected, s))
#                     if not set(succ_wires).issubset(reference_wires):
#                         continue

#                 stack.append(pred)

#             # Successors
#             for succ in G_directed.successors(current):
#                 if succ in visited:
#                     continue

#                 succ_wires = set(get_wire(G_undirected, succ))
#                 if not succ_wires.issubset(reference_wires):
#                     continue

#                 if G_directed.nodes[succ].get('num_q') == 2:
#                     preds = between_nodes(G_directed, current, succ)
#                     if len(preds) == 0:
#                         continue
#                     preds_wires = []
#                     for p in preds:
#                         preds_wires.extend(get_wire(G_undirected, p))
#                     if not set(preds_wires).issubset(reference_wires):
#                         continue

#                 stack.append(succ)

#         if len(valid_nodes) > 1:
#             subG = G_undirected.subgraph(valid_nodes).copy()
#             subgraphs.append({'center': center, 'wires': list(reference_wires), 'subG': subG})
#     return subgraphs

def get_subgraph(G_sub): 
    subgraphs = []
    G_directed = G_sub
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
            flag = False
            for pred in G_directed.predecessors(current):
                if pred in visited:
                    continue

                pred_wires = set(get_wire(G_undirected, pred))
                if not pred_wires.issubset(reference_wires):
                    flag = True # break 
                    continue

                if (flag==True) and (G_directed.nodes[pred].get('num_q') == 2):
                    continue

                stack.append(pred)

            # Successors
            flag = False
            for succ in G_directed.successors(current):
                if succ in visited:
                    continue

                succ_wires = set(get_wire(G_undirected, succ))
                if not succ_wires.issubset(reference_wires):
                    flag = True
                    continue

                if (flag==True) and (G_directed.nodes[succ].get('num_q') == 2):
                    continue

                stack.append(succ)

        if len(valid_nodes) > 1:
            subG = G_undirected.subgraph(valid_nodes).copy()
            subgraphs.append(subG)
    return subgraphs

def node_match_func(attrs1, attrs2):
    return attrs1 == attrs2

def get_unique_subgraphs(graph_list):
    unique_graphs = []
    for G_new in graph_list:
        is_unique = True
        for G_unique in unique_graphs:
            if nx.is_isomorphic(G_new, G_unique, node_match=node_match_func):
                is_unique = False
                break
        if is_unique:
            unique_graphs.append(G_new)
    return unique_graphs

# 1113
from collections import Counter
def greedy_CNOT(communities):
    mutable_blocks = [set(block) for block in communities]
    
    block_info = []
    for block in communities:
        center = None
        if any('CNOT' in gate for gate in block):
            center = 'CNOT'
        elif any('CZ' in gate for gate in block):
            center = 'CZ'
        
        block_info.append({'center': center, 'set': mutable_blocks[communities.index(block)]})
    
    all_nodes = [node for block in communities for node in block]
    node_counts = Counter(all_nodes)
    duplicated_nodes = {node for node, count in node_counts.items() if count > 1}
    
    for node in duplicated_nodes:
        is_in_cnot_block = False
        for info in block_info:
            if info['center'] == 'CNOT' and node in info['set']:
                is_in_cnot_block = True
                break
        
        if is_in_cnot_block:
            for info in block_info:
                if info['center'] == 'CZ' and node in info['set']:
                    info['set'].discard(node)

    final_blocks = []
    for block_set in mutable_blocks:
        if block_set:
            final_blocks.append(list(block_set))
            
    return final_blocks

def get_communities(G_sub):
    gate_2q = [n for n, attr in G_sub.nodes(data=True) if attr.get('num_q') == 2]
    communities = []
    for center in gate_2q:
        neighbors = set(G_sub.to_undirected().neighbors(center))
        neighbors = list(neighbors.difference(set(gate_2q)))
        if len(neighbors)<2:
            continue
        neighbors.append(center)
        communities.append(neighbors)
    return greedy_CNOT(communities)
