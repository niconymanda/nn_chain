import numpy as np

# AUX FUNCTIONS 

class LinkageUnionFind:
    """Structure for fast cluster labeling in unsorted dendrogram."""

    def __init__(self, n):
        self.parent = np.arange(2 * n - 1, dtype=np.intc)
        self.next_label = n
        self.size = np.ones(2 * n - 1, dtype=np.intc)

    def merge(self, x, y):
        self.parent[x] = self.next_label
        self.parent[y] = self.next_label
        size = self.size[x] + self.size[y]
        self.size[self.next_label] = size
        self.next_label += 1
        return size

    def find(self, x):
        p = x

        while self.parent[x] != x:
            x = self.parent[x]

        while self.parent[p] != x:
            p, self.parent[p] = self.parent[p], x

        return x

def label(Z, n):
    """Correctly label clusters in unsorted dendrogram."""
    uf = LinkageUnionFind(n)

    for i in range(n - 1):
        x, y = int(Z[i, 0]), int(Z[i, 1])
        x_root, y_root = uf.find(x), uf.find(y)
        if x_root < y_root:
            Z[i, 0], Z[i, 1] = x_root, y_root
        else:
            Z[i, 0], Z[i, 1] = y_root, x_root
        Z[i, 3] = uf.merge(x_root, y_root)

def new_dist(d_xi, d_yi, d_xy, size_x, size_y, size_i):
    t = 1.0 / (size_x + size_y + size_i)
    return np.sqrt((size_i + size_x) * t * d_xi * d_xi + (size_i + size_y) * t * d_yi * d_yi - size_i * t * d_xy * d_xy)

def condensed_index(n, i, j):
    """
    Calculate the condensed index of element (i, j) in an n x n condensed
    matrix.
    """
    if i < j:
        return n * i - (i * (i + 1) / 2) + (j - i - 1)
    elif i > j:
        return n * j - (j * (j + 1) / 2) + (i - j - 1)

# MAIN

def standard_nn_chain(D, n):

    """Calculates the NN chain algorithm w on the fly distances"""

    # Variable definition
    Z = np.empty((n - 1, 4))
    size = np.ones(n, dtype=np.intc)
    
    # Variables to store neighbors chain.
    cluster_chain = np.ndarray(n, dtype=np.intc)
    chain_length = 0

    # Begin
    for l in range(n - 1):

        # this sets the new starting cluster
        if chain_length == 0:
            chain_length = 1
            for i in range(n):
                if size[i] > 0:
                    cluster_chain[0] = i
                    break

        # This find the minimal distance and checks whether the clusters are reciprocal
        while True:
            x = cluster_chain[chain_length - 1]

            if chain_length > 1:
                y = cluster_chain[chain_length - 2]
                current_min = D[int(condensed_index(n, x, y))]
            else:
                current_min = np.inf

            for i in range(n):
                if size[i] == 0 or x == i:
                    continue
                
                dist = D[int(condensed_index(n, x, i))]
                if dist < current_min:
                    current_min = dist
                    y = i

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
        nx = size[x]
        ny = size[y]

        # Record the new node.
        Z[l, 0] = x
        Z[l, 1] = y
        Z[l, 2] = current_min
        Z[l, 3] = nx + ny
        size[x] = 0  # Cluster x will be dropped.
        size[y] = nx + ny  # Cluster y will be replaced with the new cluster

        for i in range(n):
            ni = size[i]
            if ni == 0 or i == y:
                continue

            D[int(condensed_index(n, i, y))] = new_dist(
                D[int(condensed_index(n, i, x))],
                D[int(condensed_index(n, i, y))],
                current_min, nx, ny, ni)

    order = np.argsort(Z[:, 2], kind='mergesort')
    Z_arr = Z[order]

    # Find correct cluster labels inplace.
    label(Z_arr, n)

    return Z_arr