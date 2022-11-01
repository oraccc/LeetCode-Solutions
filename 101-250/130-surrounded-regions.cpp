vector<int> directions{-1, 0, 1, 0, -1};
void solve(vector<vector<char>>& board) {
    int m = board.size(), n = board[0].size();
    vector<vector<bool>> canFlip(m, vector<bool>(n, true));
    
    for (int i = 0; i < m; ++i) {
        dfs(i, 0, board, canFlip);
        dfs(i, n-1, board, canFlip);
    }

    for (int j = 0; j < n; ++j) {
        dfs(0, j, board, canFlip);
        dfs(m-1, j, board, canFlip);
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (board[i][j] == 'O' && canFlip[i][j])
                board[i][j] = 'X';
        }
    }
}

void dfs(int x, int y, vector<vector<char>> &board, vector<vector<bool>> &canFlip) {
    if (board[x][y] == 'X' || !canFlip[x][y]) {
        return;
    }
    canFlip[x][y] = false;
    for (int k = 0; k < 4; ++k) {
        int row = x + directions[k], col = y + directions[k+1];
        if (row >= 0 && row < board.size() && col >= 0 && col < board[0].size()) {
            dfs(row, col, board, canFlip);
        }
    }
}