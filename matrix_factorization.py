from sklearn.decomposition import NMF
import numpy as np
from data_utils import *
import time

# Matrix Factorization
def matrix_factorization(matrix, n_components=2):
    nmf = NMF(n_components=n_components, init='random', random_state=0, max_iter=1000)
    W = nmf.fit_transform(matrix)  # Ma trận W
    H = nmf.components_            # Ma trận H

    community_labels = np.argmax(W, axis=1)
    communities = [set(np.where(community_labels == c)[0]) for c in range(n_components)]
    return communities

# Find the best number of communities based on modularity
def best_modularity(matrix, graph):
    best_modularity_score = -1
    best_n_components = 0
    best_communities = []
    for n_components in range(2, 10):
        communities = matrix_factorization(matrix, n_components)
        modularity_score = modularity(graph, communities)
        if modularity_score > best_modularity_score:
            best_modularity_score = modularity_score
            best_n_components = n_components
            best_communities = communities
    print("\nBest Modularity:", best_modularity_score)
    print("Best Number of Communities:", best_n_components)
    return best_n_components, best_communities

if __name__ == "__main__":
    # Read file MTX
    matrix , graph = read_mtx("soc-dolphins/soc-dolphins.mtx")

    start = time.time()
    # Find the best number of communities
    best_n_components, best_communities = best_modularity(matrix, graph)
    print("Time:", time.time() - start)

    # Show the best communitie
    show_communities(graph, best_communities)