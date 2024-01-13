int removeStones(vector<vector<int>>& stones) {
    int n = stones.size();
    unordered_map<int, vector<int>> row, col;
    for (int i = 0; i < n; ++i) {
        row[stones[i][0]].push_back(i);
        col[stones[i][1]].push_back(i);
    }

    unordered_set<int> visited;
    int connected = 0;
    for (int i = 0; i < n; ++i) {
        if (visited.count(i) == 0) {
            connected += 1;
            dfs(visited, stones, row, col, i);
        }
    }

    return n - connected; 
}

void dfs(unordered_set<int> &visited, vector<vector<int>> &stones, 
    unordered_map<int, vector<int>> &row, unordered_map<int, vector<int>> &col, int i) {
    if (visited.count(i) != 0) return;
    visited.insert(i);
    int r = stones[i][0], c = stones[i][1];
    for (const auto &s : row[r]) 
        dfs(visited, stones, row, col, s);
    for (const auto &s : col[c])
        dfs(visited, stones, row, col, s);
}