bool validPath(int n, vector<vector<int>>& edges, int source, int destination) {
    vector<vector<int>> connected(n, vector<int>());
    vector<bool> visited(n, false);

    for (const auto &e : edges) {
        connected[e[0]].push_back(e[1]);
        connected[e[1]].push_back(e[0]);
    }

    bool flag = false;
    dfs(source, destination, connected, visited, flag);

    return flag;
}

void dfs(int source, int destination, vector<vector<int>> &connected, vector<bool> &visited, bool &flag) {
    if (flag == true || visited[source] == true) return;
    if (source == destination) {
        flag = true;
        return;
    }
    visited[source] = true;
    for (const auto &node : connected[source]) {
        dfs(node, destination, connected, visited, flag);
    }
    // visited[source] = false;
}