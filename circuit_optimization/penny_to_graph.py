import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import random
import networkx as nx
import pennylane as qml
from transpilation_util import extract_info_from_qnode

class Penny_to_Graph():
    def __init__(self, qnode):
        """This class takes a QNode in PennyLane and returns a corresponding graph object in NetworkX"""
        self.qnode = qnode
        self.circuit_info = extract_info_from_qnode(self.qnode)
        # colors
        self.gate_colors = {}
        self.set_gate_colors()
        self.extra_colors = {'Qubit': '#808080', 'Measurement': "#808080"}
        self.legend_patches = [mpatches.Patch(color=color, label=label) for label, color in {**self.gate_colors, **self.extra_colors}.items()]
        # augmented and original graphs
        self.G_aug = None
        self.pos_aug = None
        self.node_colors_aug = []
        self.circuit_to_graph()
        
        self.G = None
        self.pos = None
        self.node_colors = []
        self.remove_nodes()
              
    ### functions to map gates to randomly generated colors
    @staticmethod
    def color_generator(n):
        random.seed(42)
        colors = []
        for _ in range(n):
            hex_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            colors.append(hex_color)
        return colors
    
    def set_gate_colors(self):
        """"map gates to randomly generated colors"""
        # gate name
        g_name_temp = []
        for g in self.circuit_info:
            g_name_temp.append(g['name'])
        # gate color
        n = len(g_name_temp)
        g_color_temp = ['#cc4c4c','#cc9c4c','#aacc4c','#4ccc5c','#4ccccc','#4c6ccc','#9c4ccc','#cc4ca2']
        if n <= 8:
            g_color_temp = g_color_temp[:n]
        else: # greater than 8
            g_color_temp = g_color_temp + self.color_generator(8-n)
        self.gate_colors = dict(zip(set(g_name_temp), g_color_temp))

    def show_legend(self):
        _, ax = plt.subplots()
        ax.axis('off')
        legend = ax.legend(handles=self.legend_patches, loc='center', fontsize=12, frameon=False)
        plt.tight_layout()
        plt.show()
    ###

    def circuit_to_graph(self):
        G = nx.DiGraph()
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
                    G.add_edge(timeline[i], timeline[i + 1])

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

    def show_aug_graph(self):
        _, ax = plt.subplots(figsize=(12, 6))
        nx.draw(self.G_aug, self.pos_aug, with_labels=False, node_color=self.node_colors_aug, node_size=30, font_size=8, ax=ax)
        plt.title("Graph representation of quantum circuits w/ qubits and measurements")
        plt.tight_layout()
        plt.show()

    def remove_nodes(self):
        # Step 1: Remove qubit and measurement nodes
        nodes_to_remove = [n for n, attr in self.G_aug.nodes(data=True) if attr.get('type') in ('qubit', 'measure')]
        G = self.G_aug.copy()
        G.remove_nodes_from(nodes_to_remove)

        # Step 2: Filter layout
        pos = {n: self.pos_aug[n] for n in G.nodes if n in self.pos_aug}

        # Step 3: Filter node_colors to match G_clean
        node_colors = [self.node_colors_aug[list(self.G_aug.nodes).index(n)] for n in G.nodes]

        self.G = G
        self.pos = pos
        self.node_colors = node_colors

    def show_graph(self):
        _, ax = plt.subplots(figsize=(12, 6))
        nx.draw(self.G, self.pos, with_labels=False, node_color=self.node_colors, node_size=30, font_size=8, ax=ax)
        plt.title("Graph representation of quantum circuits")
        plt.tight_layout()
        plt.show()