bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
    vector<vector<int>> graph(numCourses, vector<int>());
    vector<int> indegree(numCourses, 0);
    for (const auto& pair : prerequisites) {
        graph[pair[1]].push_back(pair[0]);
        ++indegree[pair[0]];
    }

    queue<int> q;
    for (int i = 0; i < indegree.size(); ++i) {
        if (indegree[i] == 0) {
            q.push(i);
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (auto v : graph[u]) {
            --indegree[v];
            if (indegree[v] == 0) {
                q.push(v);
            }
        }
    }

    for (int i = 0; i < indegree.size(); ++i) {
        if (indegree[i] != 0) {
            return false;
        }
    }

    return true;
}