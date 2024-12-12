import time
import networkx as nx
from networkx.algorithms.community import girvan_newman
from data_utils import *

def girvan_newman_communities(graph):
    start_time = time.time()
    comp_gen = girvan_newman(graph)
    best_modularity_score = -1
    best_communities = None

    for communities in comp_gen:
        community_list = [set(c) for c in communities]
        modularity_score = modularity(graph, community_list)

        if modularity_score > best_modularity_score:
            best_modularity_score = modularity_score
            best_communities = community_list
        else:
            break  

    print(f"Number of communities detected: {len(best_communities)}")
    end_time = time.time() 
    print(f"Time taken for Girvan-Newman community detection: {end_time - start_time:.4f} seconds")
    print(f"\nBest modularity score: {best_modularity_score}")
    return best_communities

if __name__ == "__main__":
    matrix, graph = read_mtx("soc-dolphins/soc-dolphins.mtx")
    best_communities = girvan_newman_communities(graph)
    show_communities(graph, best_communities)