import numpy as np
cimport numpy as np
import cython

@cython.infer_types(True) 
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef double ward(int size_a, int size_b, np.ndarray[double, ndim=1] pos_a, np.ndarray[double, ndim=1] pos_b):
    """calculates the ward for one cluster to another"""
    cdef int i, n = pos_a.shape[0]
    cdef double result = 0.0
    cdef double diff = 0.0

    for i in range(n):
        diff = pos_a[i] - pos_b[i]
        result += (diff) * (diff)

    return (size_a * size_b) / (size_a + size_b) * result

@cython.infer_types(True) 
@cython.wraparound(False)
cpdef get_top_k(i, size, pos, active, k):
    """Selects the top k of distances list and sorts these."""
    cdef int s = len(active)-1
    cdef np.ndarray[int, ndim=1] active_ = np.empty(s, dtype=np.int32)
    cdef np.ndarray[double, ndim=1] dists = np.empty(s, dtype=np.double)
    cdef int index = 0

    for j in active:
        if j != i:
            active_[index] = j
            dists[index] = ward(size[i], size[j], pos[i], pos[j])
            index += 1

    sorting = np.argsort(dists)[:k]
    top_k_sorted = active_[sorting]
    return top_k_sorted

@cython.infer_types(True) 
cpdef knn_chain(X, k = 5):
    """Calculates the NN chain algorithm with on the fly distances"""
    
    n = len(X)
    pos = [X[i] for i in range(n)]
    size = [1 for i in range(n)]

    active = {i for i in range(n)}
    mapping = {i: i for i in range(n)}
    reverse_mapping = {i: {i} for i in range(n)}

    chain = []
    knn = []

    dendrogram = []

    cdef int i, j, m, index, nn, new_index 
    cdef double dist_
    #cdef list knn_, dists

    # Activate loop
    while active:

        if len(active) == 2:
            i, j = active
            size_ = size[i] + size[j]
            dist_ = ward(size[i], size[j], pos[i], pos[j])
            dendrogram.append([i, j, np.sqrt(2 * dist_), size_])
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
        dendrogram.append([i, j, np.sqrt(2 * dist_), size_])
    
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