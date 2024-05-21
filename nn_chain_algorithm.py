import numpy as np

# AUX FUNCTIONS 

def ward(size_a, size_b, mu_a, mu_b):
    return (size_a * size_b) / (size_a + size_b) * np.linalg.norm( mu_a - mu_b )**2

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

# MAIN

def nn_chain(mu, k = 5):

    """Calculates the NN chain algorithm w on the fly distances"""

    # Variable definition
    n = len(mu)
    Z = np.empty((n - 1, 4))
    size = np.ones(n, dtype=np.intc)
    
    # Variables to store neighbors chain.
    cluster_chain = np.ndarray(2*n-1, dtype=np.intc)
    chain_length = 0
    knn = [np.zeros(k, dtype=int) for _ in range(2 * n - 1)]
    knn_dist = [np.zeros(k, dtype=float) for _ in range(2 * n - 1)]
    mapping = np.full((2*n-1), np.inf)
    
    # Begin
    for l in range(n - 1):

        # this sets the new starting cluster
        if chain_length == 0:
            chain_length = 1
            for i in range(2*n-1):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break

        # This find the minimal distance and checks whether the clusters are reciprocal
        while True:
            x = cluster_chain[chain_length - 1]

            # if no knn has been calculated or no knn is active
            if all(knn[x]) == 0 or all(size[knn[x]]) == 0:
                # SciPy uses Euclidean for this
                dist = wrapper_ward(x, size, mu)
                knn[x], knn_dist[x] = get_top_k(dist, k)

            y = knn[x][0]

            # if y has been merged
            if size[y] == 0:
                """
                find the first unsplit element, so that we can find out with which element it has merged.
                find out whether the order of the knn[x] has changed. 
                if any of the merged element are last in the list, we can no longer ensure these are in the top k. These are popped.
                """
                split_index = knn[x].index(0)
                merged_clusters = knn[x][:split_index] 
                unioned_clusters = np.array(list(set(mapping[merged_clusters])), dtype=int) 
                # SciPy uses Euclidean for this
                unioned_clusters_dist = [ward(size[x], size[z], mu[x], mu[z]) for z in unioned_clusters]
                
                rest = knn[x][split_index:]
                rest_dist = knn_dist[x][split_index:]

                _knn = np.append(unioned_clusters, rest)
                _dists = np.append(unioned_clusters_dist, rest_dist)
                _argsorted_dists = np.argsort(_dists)
                _i = np.where(_argsorted_dists == len(_dists) - 1)[0][0]
                reduced_dist = _argsorted_dists[:_i + 1]
                knn[x] = np.array([_knn[i] for i in reduced_dist])
                knn_dist[x] = _dists[reduced_dist]

            min_dist = knn_dist[x][0]

            # Clusters reciprocal ?
            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                break

            cluster_chain[chain_length] = y
            chain_length += 1
        
        # Merge clusters x and y and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if x > y:
            x, y = y, x

        # get the original numbers of points in clusters x and y
        size_xy = size[x] + size[y]

        # Record the new node.
        Z[l, 0] = x
        Z[l, 1] = y
        Z[l, 2] = min_dist
        Z[l, 3] = size_xy

        xy_centroid = (size[x] * mu[x] + size[y] * mu[y] ) / ( size_xy )
        mu = np.vstack([mu, xy_centroid])
        size = np.append(size, size_xy)

        size[x] = 0
        size[y] = 0

        new_index = len(mu) - 1
        mapping[x] = new_index
        mapping[y] = new_index
        mapping = np.array([new_index if m == x or m == y else m for m in mapping])

        knn[x] = [0] * k
        knn[y] = [0] * k

    return Z