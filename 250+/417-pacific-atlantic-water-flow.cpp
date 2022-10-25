vector<int> directions{-1, 0, 1, 0, -1};
vector<vector<int>> pacificAtlantic(vector<vector<int>>& heights) {
    int m = heights.size(), n = heights[0].size();
    if (m == 0 || n == 0) return {};

    vector<vector<bool>> canReachPacific(m, vector<bool>(n, false));
    vector<vector<bool>> canReachAtlantic(m, vector<bool>(n, false));

    for (int i = 0; i < m; ++i) {
        dfs(heights, canReachPacific, i, 0);
        dfs(heights, canReachAtlantic, i, n-1);
    }

    for (int j = 0; j < n; ++j) {
        dfs(heights, canReachPacific, 0, j);
        dfs(heights, canReachAtlantic, m-1, j);
    }

    vector<vector<int>> ans;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (canReachAtlantic[i][j] && canReachPacific[i][j]) {
                ans.push_back(vector<int>{i, j});
            }
        }
    }

    return ans;
}

void dfs(vector<vector<int>> &heights, vector<vector<bool>> &canReach, 
    int row, int col){
    if (canReach[row][col]) return;
    canReach[row][col] = true;
    for (int k = 0; k < 4; ++k) {
        int x = row + directions[k], y = col + directions[k+1];
        if (x >= 0 && x < heights.size() && y >= 0 && y < heights[0].size() &&
            heights[x][y] >= heights[row][col])
            dfs(heights, canReach, x, y);
    }
}