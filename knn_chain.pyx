import cython
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def ward(int size_a, int size_b, np.ndarray pos_a, np.ndarray pos_b):
    """calculates the ward for one cluster to another"""

    cdef double diff = 0.0

    for a, b in zip(pos_a, pos_b):
        diff += (a - b)**2

    return (size_a * size_b) / (size_a + size_b) * diff


def get_top_k(int i, list size, list pos, set active, int k):
    """Selects the top k of distances list and sorts these."""
    active_ = np.array([j for j in active if j != i])
    dists = [ward(size[i], size[j], pos[i], pos[j]) for j in active_]
    sorting = np.argsort(dists)[:k]
    top_k_sorted = active_[sorting]
    return top_k_sorted


def knn_chain(list X, int k = 5):
    """Calculates the NN chain algorithm with on the fly distances"""
    
    cdef int n = len(X)
    cdef list pos = [np.array(X[i]) for i in range(n)]
    cdef list size = [1 for i in range(n)]

    cdef set active = {i for i in range(n)}
    cdef dict mapping = {i: i for i in range(n)}
    cdef dict reverse_mapping = {i: {i} for i in range(n)}

    cdef list chain = []
    cdef list knn = []

    cdef list dendrogram = []

    # Activate loop
    while active:

        if len(active) == 2:
            i, j = list(active)
            size_ = size[i] + size[j]
            dist_ = ward(size[i], size[j], pos[i], pos[j])
            dendrogram.append([i, j, sqrt(2 * dist_), size_])
            return dendrogram
        
        # New chain
        if not len(chain):
            i = next(iter(active))
            chain.append(i)

            knn_ = get_top_k(i, size, pos, active, k)
            knn.append(knn_)

        while len(chain):

            i = chain[-1]
            m = -1
            for index, nn in enumerate(knn[-1]):
                if nn in active:
                    m = index
                    break

            if m <= 0:
                if m < 0:
                    knn[-1] = get_top_k(i, size, pos, active, k)
                j = knn[-1][0]
            else:
                indices = set()
                for nn in knn[-1][:m]:
                    indices |= reverse_mapping[nn]
                    
                clusters = set()
                for index in indices:
                    clusters.add(mapping[index])
                    
                knn_ = list(clusters) + [knn[-1][m]]
                dists = [ward(size[i], size[j], pos[i], pos[j]) for j in knn_]
                j = knn_[np.argmin(dists)]

            if len(chain) > 1 and chain[-2] == j:
                break

            chain.append(j)

            knn_ = get_top_k(j, size, pos, active, k)
            knn.append(knn_)

        # Merge
        dist_ = ward(size[i], size[j], pos[i], pos[j])
        size_ = size[i] + size[j]
        dendrogram.append([i, j, sqrt(2 * dist_), size_])
    
        # Update variables
        centroid = (size[i] * pos[i] + size[j] * pos[j] ) / size_
        pos.append(centroid)
        
        new_index = len(size)
        size[i] = 0
        size[j] = 0
        size.append(size_)
        
        # Update mapping
        for index in reverse_mapping[i] | reverse_mapping[j]:
            mapping[index] = new_index
            
        reverse_mapping[new_index] = reverse_mapping[i] | reverse_mapping[j]
        
        # Update active set
        active.remove(i)
        active.remove(j)
        active.add(new_index)

        chain = chain[:-2]
        knn = knn[:-2]

    return dendrogram