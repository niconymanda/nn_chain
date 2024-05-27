import numpy as np
from aux_functions import *

def test_nn_chain(X, k = 5):

    """Calculates the NN chain algorithm w on the fly distances"""

    # Variable definition
    n = len(X)
    size = np.ones(n, dtype=np.intc)

    active = {i for i, _ in enumerate(X)}
    _active = [True for _ in X]

    knn = []
    knn_dist = []
    mapping = np.full((2*n-1), np.inf)

    cluster_chain = np.empty(0, dtype=np.intc)
    chain_length = 0

    Z = np.empty((0,4), int)

    # Activate loop
    while active:

        if len(active) == 1:
            break
        
        # New chain
        if chain_length == 0:
            i = next(iter(active))
            cluster_chain = np.append(cluster_chain, i)

            chain_length = 1 # Do i still need this variable ?
            _dists = wrapper_ward(i, size, X)
            _knn, _knn_dist = get_top_k(_dists, k)
            knn.append(_knn)
                #np.array(_knn))
            knn_dist.append(_knn_dist)
                #np.array(_knn_dist))

        while cluster_chain.size:
            print("------------")
            print(f"cluster_chain = {cluster_chain}, chain_length = {chain_length} ==> {cluster_chain[chain_length-1]}")
            i = cluster_chain[chain_length - 1]
            print(chain_length > 1, cluster_chain[chain_length - 2])

            m = 0
            ind = 1

            for index, nn in enumerate(knn[-1]):
                print(f"checking active of NN = {nn}")
                if _active[nn]:
                    m = index
                    ind = 0
                    break

            print(f"m  = {m}")
            if ind:
                _dists = wrapper_ward(i, size, X, cluster_chain[chain_length - 2])
                knn[-1], knn_dist[-1] = get_top_k(_dists, k)

            if knn[-1][:m].size:
                _knn = np.array(list(set(mapping[knn[-1][:m]])), dtype=int)
                knn[-1] = np.append(_knn, knn[-1][m:])# np.array(np.append(_knn, knn[-1][m:]))
                knn_dist[-1] = np.append(dist_calculation(i, _knn, size, X), knn_dist[-1][m:])#np.array(np.append(dist_calculation(i, _knn, size, X), knn_dist[-1][m:]))

            j = knn[-1][np.argmin(knn_dist[-1])]
            print(f"i = {i}, j = {j}")
            print(f"knn = {knn[-1]}, dists = {knn_dist[-1]}")

            if chain_length > 1 and j == cluster_chain[chain_length - 2]:
                break

            cluster_chain = np.append(cluster_chain, j)
            chain_length += 1
            _dists = wrapper_ward(j, size, X, prev_element=i)
            print(_dists)
            _knn, _knn_dist = get_top_k(_dists, k)
            knn.append(_knn)
                #np.array(_knn))
            knn_dist.append(_knn_dist)
                #np.array(_knn_dist))

        print()
        print(f"merging {i, j}")
        # Merging i and j
        chain_length -= 2

        size_xy = size[i] + size[j]

        # Record the new node
        Z = np.vstack([Z, [i, j, min(knn_dist[-1]), size_xy]])

        ij_centroid = (size[i] * X[i] + size[j] * X[j] ) / ( size_xy )
        X = np.vstack([X, ij_centroid])
        print(f"new mu = {X}")
        
        size[i] = 0
        size[j] = 0
        size = np.append(size, size_xy)

        new_index = len(X) - 1

        mapping[i] = new_index
        mapping[j] = new_index
        mapping = np.array([new_index if m == i or m == j else m for m in mapping])
        print(f"mapping = {mapping}")

        active.remove(i)
        active.remove(j)
        active.add(new_index)

        _active[i] = False
        _active[j] = False
        _active = np.append(_active, True)

        cluster_chain = cluster_chain[:-2]
        knn = knn[:chain_length]
        knn_dist = knn_dist[:chain_length]

        print()
        print()

    return Z