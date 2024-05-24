import numpy as np

def ward(size_a, size_b, X_a, X_b):
    return (size_a * size_b) / (size_a + size_b) * np.sqrt( sum ( ( X_a - X_b ) **2 ) )

def wrapper_ward(i, size, X):
    """Calculates all distances from one point to the others"""
    return [ward(size[i], size[j], X[i], X[j]) if j != i and size[j] > 0 else np.inf for j, _ in enumerate(X)]

def get_top_k(distances, k):
    """Selects the top k of distances list and sorts these"""
    top_k = np.argpartition(distances, k)[:k]
    top_k_dist = np.array([distances[i] for i in top_k])

    # Re-sort the top k results
    sorted_indices = np.argsort([distances[i] for i in top_k])
    top_k_sorted = np.array(top_k[sorted_indices])
    top_k_dist_sorted = np.array(top_k_dist[sorted_indices])
    
    return top_k_sorted, top_k_dist_sorted

def dist_calculation(i, targets, size, X):
    if targets.size:
        return np.array([ward(size[i], size[t], X[i], X[t]) for t in targets])
    else:
        return np.array([])