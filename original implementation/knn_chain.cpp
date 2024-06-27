
#include <iostream>
#include <cmath>
#include<vector> 
#include <unordered_set>
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
    vector<int> active_, sorting, top_k;
    vector<double> dists;
    double ds;

    for (auto j = active.begin(); j != active.end(); ++j) {
        if (*j != i) {
            active_.push_back(*j);
            ds = ward( size[i], size[*j], pos[i], pos[*j] );
            dists.push_back( ds );
        }
    }
    for (int j = 0; j < active_.size(); j++) {
        cout << active_[j] << endl;
        cout << dists[j] << endl << endl;
    }
    sorting = argsort(dists);
    sorting.resize(k);

    for (int i = 0; i < k; i++) {
        top_k.push_back(active_[sorting[i]]);
        cout << "sorting, top_k, dist: " << endl;
        cout << sorting[i] << endl;
        cout << top_k[i] << endl;
        cout << dists[sorting[i]] << endl;
    }

    return top_k;
}

int main(){
    // TESTING WARD
    // int size_a, size_b;
    // size_a = size_b = 1;
    // vector<double> pos_a = {1.0, 2.0};
    // vector<double> pos_b = {4.0, 5.0};
    // cout << ward(size_a, size_b, pos_a, pos_b);

    // TESTING GET_TOP_K
    vector<int> size = {1,1,1,1};
    vector<vector<double>> pos = {{1.0, 2.0}, {4.0, 5.0}, {2.0, 8.0}, {2.0, 10.0}}; 
    unordered_set<int> active = {0, 1, 2, 3}; 

    vector<int> top_k = get_top_k(0, size, pos, active, 2);

    return 0;
}
