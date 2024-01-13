bool possibleBipartition(int n, vector<vector<int>>& dislikes) {
    vector<int> colors(n, -1);
    vector<vector<int>> dislikeGraph(n, vector<int>());
    for (const auto &pair : dislikes) {
        dislikeGraph[pair[0]-1].push_back(pair[1]-1);
        dislikeGraph[pair[1]-1].push_back(pair[0]-1);
    }

    for (int i = 0; i < n; ++i) {
        if (colors[i] == -1 && !dfs(i, 0, dislikeGraph, colors))
            return false;
    }

    return true;
}

bool dfs(int i, int color_id, vector<vector<int>> &dislikeGraph, vector<int> &colors) {
    if (colors[i] != -1) {
        return colors[i] == color_id;
    }

    colors[i] = color_id;
    for (const auto &each : dislikeGraph[i]) {
        if (!dfs(each, 1-color_id, dislikeGraph, colors))
            return false;
    }

    return true;
}