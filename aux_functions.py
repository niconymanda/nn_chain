import numpy as np

def ward(size_a, size_b, mu_a, mu_b):
    return (size_a * size_b) / (size_a + size_b) * np.sqrt( sum ( ( mu_a - mu_b ) **2 ) )

def wrapper_ward(x, size, mu):
    """Calculates all distances from one point to the others"""
    return [ward(size[x], size[y], mu[x], mu[y]) if y != x and size[y] > 0 else np.inf for y, _ in enumerate(mu)]

def get_top_k(distances, k):
    """Selects the top k of distances list and sorts these"""
    top_k = np.argpartition(distances, k)[:k]
    top_k_dist = np.array([distances[i] for i in top_k])

    # Re-sort the top k results
    sorted_indices = np.argsort([distances[i] for i in top_k])
    top_k_sorted = top_k[sorted_indices]
    top_k_dist_sorted = top_k_dist[sorted_indices]
    
    return top_k_sorted, top_k_dist_sorted