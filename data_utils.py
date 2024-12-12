import networkx as nx
from scipy.io import mmread
import matplotlib.pyplot as plt

# Read file MTX
def read_mtx(file_path):
    matrix = mmread(file_path)
    graph = nx.from_scipy_sparse_array(matrix)
    return matrix, graph

# Show graph
def show_graph(graph):
    pos = nx.spring_layout(graph, seed=0)
    
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos=pos, with_labels=True, node_color='skyblue', node_size=600, edge_color='gray')
    plt.title("Social Network of Dolphins")
    plt.show()

# Show communities
def show_communities(graph, communities):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph, seed=0)  # Tạo layout cho đồ thị
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(graph, pos, nodelist=community, node_size=800, node_color=f'C{i}')
    nx.draw_networkx_edges(graph, pos, alpha=0.3)
    nx.draw_networkx_labels(graph, pos)
    plt.title("Communities in Social Network of Dolphins")
    plt.show()

# Modularity
def modularity(graph, communities):
    modularity_score = nx.algorithms.community.quality.modularity(graph, communities)
    print("Number of communicaties", len(communities), "\nModularity:", modularity_score)
    return modularity_score

if __name__ == "__main__":
    matrix, graph = read_mtx("soc-dolphins/soc-dolphins.mtx")
    show_graph(graph)