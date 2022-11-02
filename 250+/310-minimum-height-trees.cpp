vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
    vector<int> ans;
    if (n == 1) {
        ans.push_back(0);
        return ans;
    }

    vector<vector<int>> graph(n, vector<int>());
    vector<int> indegree(n, 0);
    for (const auto &edge : edges) {
        graph[edge[0]].push_back(edge[1]);
        graph[edge[1]].push_back(edge[0]);
        ++indegree[edge[0]];
        ++indegree[edge[1]];
    }

    queue<int> leaves;
    int totalNodes = n;
    for (int i = 0; i < n; ++i) {
        if (indegree[i] == 1) {
            leaves.push(i);
        }
    }

    while (!leaves.empty()) {
        if (totalNodes <= 2) break;
        int leavesSize = leaves.size();
        totalNodes -= leavesSize;
        while (leavesSize--) {
            int leave = leaves.front();
            leaves.pop();
            indegree[leave] = -1;
            for (const auto &e : graph[leave]) {
                --indegree[e];
                if (indegree[e] == 1) {
                    leaves.push(e);
                }
            }
        }
    }

    while (!leaves.empty()) {
        ans.push_back(leaves.front());
        leaves.pop();
    }

    return ans;
}