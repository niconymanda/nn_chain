import numpy as np
from aux_functions import *

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
            print(f"x = {x}")
            print(f"knn of x = {knn[x], not any(knn[x])}")
            print(f"condition of size of all knns of x = {size[knn[x]], not any(size[knn[x]])}")
            # if no knn has been calculated or no knn is active
            if not any(knn[x]) or not any(size[knn[x]]):
                # SciPy uses Euclidean for this
                dist = wrapper_ward(x, size, mu)
                knn[x], knn_dist[x] = get_top_k(dist, k)
                print(f"all distances to x = {dist}")

            y = knn[x][0]
            print(f"original y = {y}")
            
            # if y has been merged
            if size[y] == 0:
                print("original y has been merged")
                """
                find the first unsplit element, so that we can find out with which element it has merged.
                find out whether the order of the knn[x] has changed. 
                if any of the merged element are last in the list, we can no longer ensure these are in the top k. These are popped.
                """
                split_index = np.where(size[knn[x]] > 0)[0][0]
                print(f"split i = {split_index}")
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

                y = knn[x][0]

            print(f"knn[x] = {knn[x]}")
            print(f"dists = {knn_dist[x]}")
            print(f"y = {y}")

            min_dist = knn_dist[x][0]

            # Clusters reciprocal ?
            if chain_length > 1 and y == cluster_chain[chain_length - 2]:
                print(f"clusters reciprocal; merging {x} and {y}")
                break

            cluster_chain[chain_length] = y
            chain_length += 1

            print()
        
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
        print(f"centroid = {xy_centroid}")
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
        print()
        print()

    return Z