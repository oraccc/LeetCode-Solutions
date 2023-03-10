class UF {
    vector<int> fa;
public:
    UF(int n): fa(n) {
        for (int i = 0; i < n; ++i) {
            fa[i] = i;
        }
    }

    int find(int p) {
        if (p == fa[p]) return p;
        else {
            fa[p] = find(fa[p]);
            return fa[p];
        }
    }

    void merge(int p, int q) {
        fa[find(p)] = find(q);
    }

    bool isConnected(int p, int q) {
        return find(p) == find(q);
    }
};


class Solution {
public:
    vector<int> findRedundantConnection(vector<vector<int>>& edges) {
        int n = edges.size();
        UF uf(n+1);
        for (const auto &e : edges) {
            int u = e[0], v = e[1];
            if (uf.isConnected(u, v)) {
                return e;
            }
            uf.merge(u, v);
        }
        return vector<int>{-1, -1};
    }
};