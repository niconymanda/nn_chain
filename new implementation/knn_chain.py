import numpy as np

class DistanceCache:
    def __init__(self):
        self.distances = {}
        self.calculated_pairs = set()

    def _make_key(self, i, j):
        if i is None or j is None:
            return False
        return (min(i, j), max(i, j))

    def add_distance(self, i, j, distance):
        key = self._make_key(i, j)
        self.distances[key] = distance
        self.calculated_pairs.add(key)

    def get_distance(self, i, j):
        key = self._make_key(i, j)
        return self.distances.get(key, None)

    def exists(self, i, j):
        key = self._make_key(i, j)
        return key in self.calculated_pairs
    
    def remove_distance(self, i, j):
        key =  self._make_key(i, j)
        self.distances.pop(key, None)
        self.calculated_pairs.remove(key)


def ward(size_a, size_b, pos_a, pos_b):
    """calculates the ward for one cluster to another"""
    n = pos_a.shape[0]
    result = 0.0
    diff = 0.0
    
    for i in range(n):
        diff = pos_a[i] - pos_b[i]
        result += (diff) * (diff)
    
    return np.sqrt( 2 * ((size_a * size_b) / (size_a + size_b) * result) )


def knn_pdist(X):
    """calculates the ward for all nodes at the beginning"""
    
    n = len(X)
    distances = DistanceCache()
    size = np.ones(n)
    
    for i, pos_a in enumerate(X):
        for j, pos_b in enumerate(X):
            if i != j:
                if not distances.exists(i, j):
                    distance = ward(size[i], size[j], pos_a, pos_b)
                    distances.add_distance(i, j, distance)
    
    return distances


def distance_ward(size_s, size_t, size_v, d_sv, d_tv, d_st):
    """calculates the ward for one cluster to another"""
    t = 1.0 / (size_v + size_s + size_t)
    return np.sqrt( (size_v + size_s) * t * d_sv * d_sv +
                   (size_v + size_t) * t * d_tv * d_tv - 
                   size_v * t * d_st * d_st )


def get_top_k(i, distance_cache, active, recent_mapping, size, k):
    """returns top k elems of i"""
    top_k_dists = []
    top_k = []
    
    for j in active:
        if i != j:
            
            dist = distance_cache.get_distance(i, j)

            if len(top_k) < k:
                top_k_dists.append(dist)
                sorted_i = np.argsort(top_k_dists)
                top_k_dists = [top_k_dists[i] for i in sorted_i]
                
                top_k.append(j)
                top_k = [top_k[i] for i in sorted_i]
            
            else:
                if dist < top_k_dists[-1]:
                    top_k_dists[-1] = dist
                    top_k[-1] = j

                    sorted_i = np.argsort(top_k_dists)
                    top_k_dists = [top_k_dists[i] for i in sorted_i]
                                    
                    top_k = [top_k[i] for i in sorted_i]
    
    return top_k


def aux_update_distance_cache(s, t, j, i, size, distance_cache):
    """
    s, t are the previous elements that made up j
    i is a node in the chain
    """
    d_si = distance_cache.get_distance(s, i)
    d_ti = distance_cache.get_distance(t, i)
    d_st = distance_cache.get_distance(s, t)
    dist = distance_ward(size[s], size[t], size[i], d_si, d_ti, d_st)
    distance_cache.add_distance(i, j, dist)
    # distance_cache.remove_distance(s, i)
    # distance_cache.remove_distance(t, i)


def update_distance_cache(i, distance_cache, active, size, recent_mapping):
    """Case where i has been merged -> check whether j has been merged, calculate distance and update cache"""
    s,t = (None,None)
    
    if i in recent_mapping:
        s,t = recent_mapping[i]
    
    for j in active:
        if i != j:
            if not distance_cache.exists(i, j):
                
                if not (distance_cache.exists(s, j) and distance_cache.exists(t, j)):
                    q, r = recent_mapping[j]
                    aux_update_distance_cache(q, r, j, i, size, distance_cache)

                else:                    
                    aux_update_distance_cache(s, t, i, j, size, distance_cache)


def knn_chain(distance_cache, n, k):

    size = np.ones(n)

    chain = []
    knn = []

    active = {i for i in range(n)}

    mapping = {i: i for i in range(n)}
    reverse_mapping = {i: {i} for i in range(n)}
    recent_mapping = {}
    dendrogram = []

    while active:
        
        if len(active) == 2:
            i, j = active
            size_ = size[i] + size[j]
            update_distance_cache(i, distance_cache, active, size, recent_mapping)
            dist_ = distance_cache.get_distance(i, j)
            dendrogram.append([i, j, dist_, size_])
            
            return dendrogram
        
        if not chain:
            i = next(iter(active))
            chain.append(i)
            update_distance_cache(i, distance_cache, active, size, recent_mapping)
            knn_ = get_top_k(i, distance_cache, active, recent_mapping, size, k)
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
                    knn[-1] = get_top_k(i, distance_cache, active, recent_mapping, size, k)
                j = knn[-1][0]
            
            else:
                indices = set()
                for nn in knn[-1][:m]:
                    indices |= reverse_mapping[nn]
                    
                clusters = set()
                for index in indices:
                    clusters.add(mapping[index])
                    
                knn_ = list(clusters) + [knn[-1][m]]
                dists = [distance_cache.get_distance(i,j) for j in knn_]
                j = knn_[np.argmin(dists)]
            
            if len(chain) > 1 and chain[-2] == j:
                break
            
            chain.append(j)
            update_distance_cache(j, distance_cache, active, size, recent_mapping)
            knn_ = get_top_k(j, distance_cache, active, recent_mapping, size, k)
            knn.append(knn_)

        # Merging
        dist_ = distance_cache.get_distance(i, j)
        size_ij = size[i] + size[j]
        dendrogram.append([i, j, dist_, size_ij])

        new_index = len(size)
        
        # Update mapping
        for index in reverse_mapping[i] | reverse_mapping[j]:
            mapping[index] = new_index
            
        reverse_mapping[new_index] = reverse_mapping[i] | reverse_mapping[j]
        recent_mapping[new_index] = {i, j}
        
        # Update active set
        active.remove(i)
        active.remove(j)
        active.add(new_index)

        chain = chain[:-2]
        knn = knn[:-2]

        # Update distance cache, calculate distance to previous chain elements
        update_distance_cache(new_index, distance_cache, chain, size, recent_mapping)
        size = np.append(size, size_ij)

    return dendrogram