from qco_level_0 import *
from gate_determination import *
import networkx as nx
from qiskit.transpiler.passes.synthesis import UnitarySynthesis
from qiskit.transpiler import PassManager
from qiskit import QuantumCircuit
from qco_spec_table import make_spec_table, show_circuit

def optimization_level_3_qiskit(qnode):
    G, circuit_info = optimization_prep(qnode)  
    G, communities = graph_alg_level_3(G, barriers=['QubitUnitary']) # working on
    qnode_q3 = subcircuit_syntehsis_level_3_qiskit(G, communities, circuit_info)
    print(make_spec_table(qnode, qnode_q3))
    show_circuit(qnode_q3)
    return qnode

def graph_alg_level_3(G, barriers):
    barriers = [node for node in G.nodes if any(gate in node for gate in barriers)]
    barrier_set = set(barriers)

    G_sub = G.subgraph([node for node in G.nodes if node not in barrier_set])
    communities = get_communities(G_sub)

    barriers = [n for n in list(G.nodes()) if n not in set().union(*communities)]
    for barrier in barriers:
        communities.append({barrier})

    G, communities = community_sort(G, communities, barriers)
    return community_topological_sort(G, communities)

def subcircuit_syntehsis_level_3_qiskit(G, communities, circuit_info):
    community_circuit_info = subcircuit_syntehsis_prep(G, communities, circuit_info)

    for idx, ci in enumerate(community_circuit_info):
        if len(ci) > 1:
            matrix, wires = info_to_qnode_matrix(ci)
            community_circuit_info[idx] = wire_mapping(qiskit_optimization(matrix), wires)

    community_circuit_info = [item[0] for item in community_circuit_info if isinstance(item, list) and item and isinstance(item[0], dict)]
    community_circuit_info = [gate for gate in community_circuit_info if gate]

    return info_to_qnode(community_circuit_info)

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

def get_communities(G_sub):
    subgraphs = get_subgraph(G_sub)
    unique_subgraphs = get_unique_subgraphs(subgraphs)

    communities = []
    for elem in unique_subgraphs:
        communities.append(set(elem['nodes']))
    return communities

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

def wire_mapping(circuit_info, wires):
    wire_map = {0: wires[0], 1: wires[1]}
    remapped_gates = []
    for gate in circuit_info:
        remapped_gate = gate.copy()
        remapped_gate['wires'] = [wire_map[w] for w in gate['wires']]
        remapped_gates.append(remapped_gate)
    return remapped_gates

def qiskit_optimization(u):
    basis = ['h', 'x', 'cx', 'cz', 'ry', 'u', 'u1', 'u2'] 
    qc = QuantumCircuit(2)
    qc.unitary(u, range(2))
    synth_pass = UnitarySynthesis(basis_gates=basis) # basis_gates=basis
    pm = PassManager(synth_pass)
    optimized_circuit = pm.run(qc)
    qnoe_temp = qml.from_qiskit(optimized_circuit)
    dev = qml.device("default.qubit", wires=2)
    return extract_info_from_qnode(qml.QNode(qnoe_temp, dev))

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

