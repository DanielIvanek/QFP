import matplotlib.pyplot as plt
import networkx as nx

# Function to generate a binomial tree
def generate_binomial_tree(periods):
    G = nx.Graph()
    for i in range(periods + 1):
        for j in range(i + 1):
            G.add_node((i, j), value=0)  # Adding nodes with initial values

    for i in range(periods):
        for j in range(i + 1):
            G.add_edge((i, j), (i + 1, j), weight='1-p')  # Connecting nodes horizontally with 'p' for up movement
            G.add_edge((i, j), (i + 1, j + 1), weight='p')  # Connecting nodes diagonally with 'p-1' for down movement

    return G

# Function to plot the horizontally flipped binomial tree
def plot_binomial_tree(G):
    pos = {(i, j): (i+4, 4-(i-j)) for i, j in G.nodes()}  # Adjusted the pos dictionary for horizontal flip
    labels = {(i, j): f'S_{i},{j}' for i, j in G.nodes()}  # Added time step and up movement count to node labels

    edge_labels = {edge: G.edges[edge]['weight'] for edge in G.edges()}
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=1000, node_color='lightblue', font_size=8, horizontalalignment='center')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title('Binomial Tree Example')
    plt.show()

# Example usage
periods = 4
binomial_tree = generate_binomial_tree(periods)
plot_binomial_tree(binomial_tree)
