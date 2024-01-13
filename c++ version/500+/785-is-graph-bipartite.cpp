bool isBipartite(vector<vector<int>>& graph) {
    int n = graph.size();
    vector<int> colors(n, 0);
    queue<int> q;
    for (int i = 0; i < n; ++i) {
        if (colors[i] == 0) {
            colors[i] = 1;
            q.push(i);
        }
        while (!q.empty()) {
            int s = q.size();
            for (int j = 0; j < s; ++j) {
                int node = q.front();
                q.pop();
                for (const auto& e: graph[node]) {
                    if (colors[e] == 0) {
                        colors[e] = colors[node] == 1 ? 2:1;
                        q.push(e);
                    }
                    else if (colors[e] == colors[node]) return false;
                }
            }
        }
    }

    return true;

}