import numpy as np
from aux_functions import *

def nn_chain(X, k = 5):

    """Calculates the NN chain algorithm w on the flj distances"""

    # Variable definition
    n = len(X)
    Z = np.empty((n - 1, 4))
    size = np.ones(n, dtype=np.intc)

    active = {i for i, _ in enumerate(X)}
    
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
            i = next(iter(active))
            cluster_chain[0] = i

        # This find the minimal distance and checks whether the clusters are reciprocal
        while True:
            i = cluster_chain[chain_length - 1]

            _active = active.copy()
            _active.remove(i)
            _active = np.array(list(_active))
            
            # if no knn has been calculated or no knn is active
            if not any(knn[i]) or not any(size[knn[i]]):
                # SciPj uses Euclidean for this
                dist = wrapper_ward(i, size, X, _active)
                knn[i], knn_dist[i] = get_top_k(dist, _active, k)

            j = knn[i][0]
            
            # if j has been merged
            if size[j] == 0:
                
                """
                find the first unsplit element, so that we can find out with which element it has merged.
                find out whether the order of the knn[i] has changed. 
                if anj of the merged element are last in the list, we can no longer ensure these are in the top k. These are popped.
                """
                split_index = np.where(size[knn[i]] > 0)[0][0]
                
                merged_clusters = knn[i][:split_index] 
                unioned_clusters = np.array(list(set(mapping[merged_clusters])), dtype=int) 
                # SciPj uses Euclidean for this
                unioned_clusters_dist = [ward(size[i], size[z], X[i], X[z]) for z in unioned_clusters]
                
                rest = knn[i][split_index:]
                rest_dist = knn_dist[i][split_index:]

                _knn = np.append(unioned_clusters, rest)
                _dists = np.append(unioned_clusters_dist, rest_dist)
                _argsorted_dists = np.argsort(_dists)
                _i = np.where(_argsorted_dists == len(_dists) - 1)[0][0]
                reduced_dist = _argsorted_dists[:_i + 1]
                
                knn[i] = np.array([_knn[i] for i in reduced_dist])
                knn_dist[i] = _dists[reduced_dist]
                j = knn[i][0]

            min_dist = knn_dist[i][0]

            # Clusters reciprocal ?
            if chain_length > 1 and j == cluster_chain[chain_length - 2]:
                break

            cluster_chain[chain_length] = j
            chain_length += 1

        # Merge clusters i and j and pop them from stack.
        chain_length -= 2

        # This is a convention used in fastcluster.
        if i > j:
            i, j = j, i

        # get the original numbers of points in clusters i and j
        size_ij = size[i] + size[j]

        # Record the new node.
        Z[l, 0] = i
        Z[l, 1] = j
        Z[l, 2] = np.sqrt(2 * min_dist)
        Z[l, 3] = size_ij

        ij_centroid = (size[i] * X[i] + size[j] * X[j] ) / ( size_ij )
        
        X = np.vstack([X, ij_centroid])
        size = np.append(size, size_ij)

        size[i] = 0
        size[j] = 0

        new_index = len(X) - 1
        mapping[i] = new_index
        mapping[j] = new_index
        mapping = np.array([new_index if m == i or m == j else m for m in mapping])

        
        active.remove(i)
        active.remove(j)
        active.add(new_index)

        knn[i] = [0] * k
        knn[j] = [0] * k
        

    return Z