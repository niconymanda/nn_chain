import numpy as np

def ward(size_a, size_b, X_a, X_b):
    """calculates the ward for one cluster to another"""
    return (size_a * size_b) / (size_a + size_b) * sum ( ( X_a - X_b ) **2 )


def wrapper_ward(i, size, X, active, prev_element = None):
    """Calculates all distances from one point to all other active nodes"""
    return np.array([ward(size[i], size[j], X[i], X[j]) - 1e-5 if j == prev_element
            else ward(size[i], size[j], X[i], X[j]) for j in active])


def get_top_k(distances, active, k):
    """Selects the top k of distances list and sorts these"""
    if len(active) <= k:
        sorting = np.argsort(distances)
        top_k_sorted = active[sorting]
        top_k_dist_sorted = distances[sorting]

    else:
        top_k = np.argpartition(distances, k)[:k]
        top_k_dist = np.array(distances[top_k])
        # Re-sort the top k results
        sorted_indices = np.argsort(top_k_dist)
        top_k_sorted = active[np.array(top_k[sorted_indices])]
        top_k_dist_sorted = np.array(top_k_dist[sorted_indices])
    
    return top_k_sorted, top_k_dist_sorted


def dist_calculation(i, targets, size, X):
    """calculates new ward for new mapping"""
    if targets.size:
        return np.array([ward(size[i], size[t], X[i], X[t]) for t in targets])
    else:
        return np.array([])