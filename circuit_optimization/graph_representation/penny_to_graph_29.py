import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from penny_qiskit_utils import extract_info_from_qnode
from collections import defaultdict

class Penny_to_Graph():
    def __init__(self, qnode):
        """This class takes a QNode in PennyLane and returns a corresponding graph object in NetworkX"""
        self.qnode = qnode
        self.circuit_info = extract_info_from_qnode(self.qnode)
        # colors
        self.gate_colors = {'Hadamard':'#09a2ec','PauliX':'#d66dc3','U1':"#dddc7e",'U2':'#b4d8ae','RY':'#10b287','CZ:':'#f03028','CNOT':'#b06d67','QubitUnitary':'#6c024f'}
        # self.set_gate_colors()
        self.extra_colors = {'Qubit': '#808080', 'Measurement': "#808080"}
        self.legend_patches = [mpatches.Patch(color=color, label=label) for label, color in {**self.gate_colors, **self.extra_colors}.items()]
        
        # graph w/o qubits and measurements
        self.G_aug = None
        self.pos_aug = None
        self.node_colors_aug = []
        self.circuit_to_graph()

        self.G = None
        self.pos = None
        self.node_colors = []
        self.remove_qm_nodes()

        # graph optimization
        self.G_1q = None
        self.pos_1q = None    
        self.G_2q = None
        self.pos_2q = None    

    # core routine for the transformation
    def circuit_to_graph(self):
        G = nx.MultiDiGraph() # for multiple edges between two nodes
        num_qubits = max(w for gate in self.circuit_info for w in gate['wires']) + 1
        qubit_timelines = {q: [] for q in range(num_qubits)}

        for q in range(num_qubits):
            G.add_node(f"q[{q}]", type='qubit')

        for i, gate in enumerate(self.circuit_info):
            gate_id = f"{gate['name']}_{i}"
            G.add_node(gate_id, type='gate', label=gate['name'], params=gate['params'], wires=gate['wires'])
            for wire in gate['wires']:
                qubit_timelines[wire].append(gate_id)

        for q, timeline in qubit_timelines.items():
            if timeline:
                G.add_edge(f"q[{q}]", timeline[0])
                for i in range(len(timeline) - 1):
                    src = timeline[i]
                    tgt = timeline[i + 1]
                    wire = q  # this is the qubit wire connecting them
                    G.add_edge(src, tgt, key=f"wire_{wire}", wire=wire)

        pos = {}
        x_spacing = 0.6
        y_spacing = 0.8

        for q in range(num_qubits):
            pos[f"q[{q}]"] = (0, -q * y_spacing)

        for i, gate in enumerate(self.circuit_info):
            gate_id = f"{gate['name']}_{i}"
            avg_y = -sum(gate['wires']) / len(gate['wires']) * y_spacing
            pos[gate_id] = ((i + 1) * x_spacing, avg_y)

        max_gate_x = (len(self.circuit_info) + 1) * x_spacing
        for q, timeline in qubit_timelines.items():
            meas_id = f"Measure[{q}]"
            G.add_node(meas_id, type='measure')
            if timeline:
                G.add_edge(timeline[-1], meas_id)
            else:
                G.add_edge(f"q[{q}]", meas_id)
            pos[meas_id] = (max_gate_x, -q * y_spacing)

        node_colors = []
        for node in G.nodes:
            node_type = G.nodes[node]['type']
            if node_type == 'gate':
                node_colors.append(self.gate_colors.get(G.nodes[node]['label'], 'lightgray'))
            else:
                node_colors.append('#808080')  # qubit or measure

        self.G_aug = G
        self.pos_aug = pos
        self.node_colors_aug = node_colors

    # visualization
    def show_graph(self, input_G):
        graph_map = {
            self.G_aug: (self.pos_aug, self.node_colors_aug),
            self.G: (self.pos, self.node_colors),
            self.G_1q: (self.pos_1q, self.node_colors),  
            self.G_2q: (self.pos_2q, self.node_colors),  
        }
        pos, node_colors = graph_map.get(input_G, (None, None))

        _, ax = plt.subplots(figsize=(12, 6))
        nx.draw_networkx_nodes(input_G, pos, node_color=node_colors, node_size=30, ax=ax)
        edge_counts = defaultdict(int)
        for u, v, k, d in input_G.edges(keys=True, data=True):
            pair = (u, v)
            edge_counts[pair] += 1
            count = edge_counts[pair]
            rad = 0 if count == 1 else 0.2 * ((-1) ** count)
            nx.draw_networkx_edges(input_G, pos, edgelist=[(u, v)],
                                connectionstyle=f'arc3,rad={rad}', ax=ax, arrows=True)
        ax.legend(handles=self.legend_patches, loc='upper right', fontsize='small', frameon=False, bbox_to_anchor=(1.1, 1))
        ax.axis('off')
        plt.tight_layout()
        plt.show()

    def remove_qm_nodes(self):
        nodes_to_remove = [n for n, attr in self.G_aug.nodes(data=True) if attr.get('type') in ('qubit', 'measure')]
        G = self.G_aug.copy()
        G.remove_nodes_from(nodes_to_remove)

        pos = {n: self.pos_aug[n] for n in G.nodes if n in self.pos_aug}
        node_colors = [self.node_colors_aug[list(self.G_aug.nodes).index(n)] for n in G.nodes]

        self.G = G
        self.pos = pos
        self.node_colors = node_colors
