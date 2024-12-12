import time
import networkx as nx
import community as community_louvain
from data_utils import *

def louvain_communities(graph):
    start_time = time.time()
    # Compute the best partition using Louvain method
    partition = community_louvain.best_partition(graph)

    # Group nodes by their community
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)

    community_list = [set(nodes) for nodes in communities.values()]

    modularity_score = modularity(graph, community_list)
    print(f"Number of communities detected: {len(community_list)}")
    end_time = time.time()
    print(f"Time taken for Louvain community detection: {end_time - start_time:.4f} seconds")
    print(f"\nModularity score: {modularity_score}")

    return community_list

if __name__ == "__main__":
    matrix, graph = read_mtx("soc-dolphins/soc-dolphins.mtx")
    best_communities = louvain_communities(graph)
    show_communities(graph, best_communities)
