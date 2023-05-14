class Solution {
    vector<int> directions{-1, 0, 1, 0, -1};
public:
    int movingCount(int m, int n, int k) {
        vector<vector<bool>> visited(m, vector<bool>(n, false));
        return dfs(0, 0, m, n, k, visited);
    }

    int dfs(int i, int j, int m, int n, int k, vector<vector<bool>>& visited) {
        if (visited[i][j] || (i % 10 + i / 10 + j % 10 + j / 10) > k) return 0;
        visited[i][j] = true;
        int count = 1;
        for (int l = 0; l < 4; ++l) {
            int row = i + directions[l];
            int col = j + directions[l+1];
            if (row >= 0 && row < m && col >= 0 && col < n) {
                count += dfs(row, col, m, n, k, visited);
            }
        }

        return count;
    }
};