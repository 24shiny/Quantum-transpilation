import numpy as np
import networkx as nx
import pennylane as qml

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

class Penny_to_Graph():
    def __init__(self, qnode):
        self.qnode = qnode
        self.circuit_info = extract_info_from_qnode(self.qnode)       
        self.G = None
        self.pos = None
        self.node_colors = []
        self.circuit_to_graph()

    def circuit_to_graph(self):
        G = nx.MultiDiGraph()
        num_qubits = max(w for gate in self.circuit_info for w in gate['wires']) + 1
        qubit_timelines = {q: [] for q in range(num_qubits)}

        # Add gate nodes and build timelines
        for i, gate in enumerate(self.circuit_info):
            gate_id = f"{gate['name']}_{i}"
            G.add_node(gate_id, type='gate', label=gate['name'], params=gate['params'], wires=gate['wires'])
            for wire in gate['wires']:
                qubit_timelines[wire].append(gate_id)

        # Connect gates along each qubit's timeline
        for q, timeline in qubit_timelines.items():
            for i in range(len(timeline) - 1):
                G.add_edge(timeline[i], timeline[i + 1], key=f"wire_{q}", wire=q)

        # Annotate gate nodes with number of qubits
        for n, attr in G.nodes(data=True):
            G.nodes[n]['num_q'] = len(attr.get('wires', []))

        self.G = G
