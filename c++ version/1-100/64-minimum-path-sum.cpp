//backtracking: TLE
int minPathSum(vector<vector<int>>& grid) {
    int minSum = INT_MAX, sum = 0;
    backtracking(0, 0, grid, sum, minSum);
    return minSum;
}

void backtracking(int i, int j, vector<vector<int>> &grid, int &sum, int &minSum) {
    if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size()) return;
    if (i == grid.size() - 1 && j == grid[0].size() - 1) {
        sum += grid[i][j];
        minSum = min(minSum, sum);
        sum -= grid[i][j];
        return;
    }
    sum += grid[i][j];
    backtracking(i+1, j, grid, sum, minSum);
    backtracking(i, j+1, grid, sum, minSum);
    sum -= grid[i][j];
}

//dp
int minPathSum(vector<vector<int>>& grid) {
    int m = grid.size(), n = grid[0].size();
    vector<vector<int>> dp(m, vector<int>(n, 0));
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == 0 && j == 0) {
                dp[i][j] = grid[i][j];
            }
            else if (i == 0) {
                dp[i][j] = dp[i][j-1] + grid[i][j];
            }
            else if (j == 0) {
                dp[i][j] = dp[i-1][j] + grid[i][j];
            }
            else {
                dp[i][j] = min(dp[i-1][j], dp[i][j-1]) + grid[i][j];
            }
        }
    } 

    return dp[m-1][n-1];
}