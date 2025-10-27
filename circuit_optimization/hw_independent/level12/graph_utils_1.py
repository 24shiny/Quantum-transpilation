import matplotlib.pyplot as plt
import networkx as nx
import random

def graph_info(G):
    for node, attr in G.nodes(data=True):
        print(node, attr)

def remove_nodes(G_input, node_to_remove, by_label=False, pos_input=None, node_colors_input=None):
    """removes nodes and updates position and node colors"""
    if pos_input is None:
        pos_input = {}
    if node_colors_input is None:
        node_colors_input = {}

    # determine mode
    if by_label:
        nodes_to_remove = [n for n, attr in G_input.nodes(data=True) if attr.get('label') in node_to_remove]
    else:
        nodes_to_remove = [n for n in node_to_remove if n in G_input]

    G = G_input.copy()
    G.remove_nodes_from(nodes_to_remove)

    # Update pos and node_colors
    pos = {n: pos_input[n] for n in G.nodes if n in pos_input}
    if isinstance(node_colors_input, dict):
        node_colors = [node_colors_input[n] for n in G.nodes if n in node_colors_input]
    else:
        node_to_color = {n: node_colors_input[i] for i, n in enumerate(G_input.nodes)}
        node_colors = [node_to_color[n] for n in G.nodes if n in node_to_color]

    return [G, pos, node_colors]

def community_graph(G, pos, communities):
    def color_generator(n):
        random.seed(42)
        colors = []
        for _ in range(n):
            hex_color = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            colors.append(hex_color)
        return colors

    palette = color_generator(len(communities))
    node_color_map = {}
    for i, community in enumerate(communities):
        color = palette[i % len(palette)]
        for node in community:
            node_color_map[node] = color

    node_colors = [node_color_map.get(node, '#999999') for node in G.nodes]

    # Step 4: Draw the graph
    fig, ax = plt.subplots(figsize=(12, 6))
    nx.draw(G, pos, with_labels=False, node_color=node_colors, node_size=50, font_size=8, edge_color='gray', ax=ax)
    plt.title('Community graph')
    plt.tight_layout()
    plt.show()

def draw_subgraphs(subgraphs):
    import matplotlib.pyplot as plt
    num_subgraphs = len(subgraphs)
    cols = 5
    rows = (num_subgraphs + cols - 1) // cols 

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  
    axes = axes.flatten()  

    for i in range(num_subgraphs):
        nx.draw(subgraphs[i]['subG'], ax=axes[i], with_labels=True)
        axes[i].set_title(subgraphs[i]['center'])

    # Hide any unused axes
    for j in range(num_subgraphs, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()