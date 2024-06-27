#include <iostream>
#include <cmath>
#include<vector> 
#include <unordered_set>
#include <unordered_map>
#include <bits/stdc++.h>
using namespace std;
 
vector<int> argsort(vector<double> dists) {
    /*argsort the distances*/
    int n = dists.size();
    vector<pair<double, int>> sorting;
    vector<int> indices;


    for (int i = 0; i < n; ++i) {
        sorting.push_back(make_pair(dists[i], i));
    }

    sort(sorting.begin(), sorting.end());

    for (int i = 0; i < n ; i++) {
        indices.push_back(sorting[i].second);
    }

    return indices;
}

int argmin(vector<double> dists) {
    auto min_it = min_element(dists.begin(), dists.end());
    int min_i = distance(dists.begin(), min_it);
    return min_i;
}


double ward(int size_a, int size_b, vector<double> pos_a, vector<double> pos_b) {
    /*calculates the ward for one cluster to another*/
    int i, n = pos_a.size();
    double result, diff, s;
    result = diff = s = 0.0;

    for (i = 0; i < n; i++) {
        diff = pos_a[i] - pos_b[i];
        result += (diff) * (diff);
    }
    s = static_cast<double>(size_a * size_b) / (size_a + size_b);

    return s * result;
}

vector<int> get_top_k(int i, vector<int> size, vector<vector<double>> pos, unordered_set<int> active, int k) {
    vector<int> active_, sorting, top_k, index;
    vector<double> dists;
    double ds;

    for (auto j = active.begin(); j != active.end(); ++j) {
        if (*j != i) {
            active_.push_back(*j);
            ds = ward( size[i], size[*j], pos[i], pos[*j] );
            dists.push_back( ds );
        }
    }
    // for (int j = 0; j < active_.size(); j++) {
    //     cout << active_[j] << endl;
    //     cout << dists[j] << endl << endl;
    // }
    sorting = argsort(dists);
    sorting.resize(k);

    for (int index = 0; index < k; index++) {
        top_k.push_back(active_[sorting[index]]);
        // cout << "sorting, top_k, dist: " << endl;
        // cout << sorting[i] << endl;
        // cout << top_k[i] << endl;
        // cout << dists[sorting[i]] << endl;
    }

    return top_k;
}

vector<vector<double>> knn_chain(vector<vector<double>> X, int k = 5) {
    /*Calculates the NN chain algorithm with on the fly distances*/
    // Variable declaration & definition
    int i, j, m, index, m_index, new_index, tmp_size, n = X.size();
    double tmp_dist;
    vector<vector<double>> dendrogram, pos = X;
    vector<int> size, chain, tmp_knn;
    vector<double> dists, centroid;
    vector<vector<int>> knn;
    unordered_set<int> active, tmp_rev_mapping;
    unordered_map<int, int> mapping;
    unordered_map<int, unordered_set<int>> reverse_mapping;
    for (int i = 0; i < n; i++) {
        size.push_back(1);
        mapping[i] = i;
        reverse_mapping[i] = {i};
        active.insert(i);
    }

    while (not active.empty()) {
        // Merge the remaining two clusters
        if (active.size() == 2) {
            auto it = active.begin();
            int i = *it;
            cout << i << endl;
            ++it;
            int j = *it;
            tmp_size = size[i] + size[j];
            tmp_dist = sqrt(2 * ward(size[i], size[j], pos[i], pos[j]) );
            dendrogram.push_back({static_cast<double>(i), static_cast<double>(j), tmp_dist, static_cast<double>(tmp_size)});
            return dendrogram;
        }
        // Start new chain
        if (!chain.size()) {
            i = *active.begin();
            chain.push_back(i);
            tmp_knn = get_top_k(i, size, pos, active, k);
            knn.push_back(tmp_knn);
        }
        // Continue chain
        while (chain.size()) {
            i = chain.back();
            tmp_knn = knn.back();
            m = -1;

            for (index = 0; index < tmp_knn.size(); index++) {
                if (active.find(tmp_knn[index]) == active.end()) {
                    m = index;
                    break;
                }
            }
            if (m <= 0) {
                if (m < 0) {
                    tmp_knn = get_top_k(i, size, pos, active, k);
                }
                j = tmp_knn[0];
                knn.back() = tmp_knn;
            }
            else {
                unordered_set<int> indices;
                for (index = 0; index < m; index++) {
                    tmp_rev_mapping = reverse_mapping[tmp_knn[index]];
                    indices.insert(tmp_rev_mapping.begin(), tmp_rev_mapping.end());
                }
                unordered_set<int> clusters;
                for (int index : indices) {
                    clusters.insert(mapping[index]);
                }
                clusters.insert(tmp_knn[m]);
                for (auto index = clusters.begin(); index != clusters.end(); ++index) {
                    tmp_dist = ward(size[i], size[*index], pos[i], pos[*index]);
                    dists.push_back(tmp_dist);
                    auto it = next(clusters.begin(), argmin(dists));
                    j = *it;
                }
            }
            if (chain.size() > 1 && chain[chain.size()-2] == j) {
                break;
            }
            chain.push_back(j);
            tmp_knn = get_top_k(j, size, pos, active, k);
            knn.push_back(tmp_knn);
        }
        // Merging i, j
        tmp_dist = ward(size[i], size[j], pos[i], pos[j]);
        tmp_size = size[i] + size[j];
        dendrogram.push_back({static_cast<double>(i), static_cast<double>(j), tmp_dist, static_cast<double>(tmp_size)});

        // Update Variables
        centroid = (size[i] * pos[i] + size[j] * pos[j] ) / tmp_size; //Loop
        pos.push_back(centroid);
        new_index = size.size();
        size.push_back(tmp_size);

        // Update Mapping


    }
    

    cout << active.empty() << endl;

    return pos;
}

int main(){
    // TESTING WARD
    // int size_a, size_b;
    // size_a = size_b = 1;
    // vector<double> pos_a = {1.0, 2.0};
    // vector<double> pos_b = {4.0, 5.0};
    // cout << ward(size_a, size_b, pos_a, pos_b);

    // TESTING GET_TOP_K
    // vector<int> size = {1,1,1,1};
    // vector<vector<double>> pos = {{1.0, 2.0}, {4.0, 5.0}, {2.0, 8.0}, {2.0, 10.0}};
    // unordered_set<int> active = {0, 1, 2, 3};
    // vector<int> top_k = get_top_k(0, size, pos, active, 2);

    // TESTING KNN_CHAIN
    // vector<vector<double>> pos = {{1.0, 2.0}, {4.0, 5.0}};
    // vector<vector<double>> post = knn_chain(pos);
    // for (double val : post[0]) {
    //     cout << val << " ";
    // }
    // cout << endl;

    cout << argmin({2,3,0,5,6,0}) << endl;

    return 0;
}
