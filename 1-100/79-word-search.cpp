vector<int> directions{-1, 0, 1, 0, -1};

bool exist(vector<vector<char>>& board, string word) {
    int m = board.size(), n = board[0].size();
    vector<vector<bool>> visited(m, vector<bool>(n, false));
    int wordPos = 0;
    bool find = false;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            backtracking(i, j, board, word, visited, find, wordPos);
            if (find) return true;
        }
    }
    return false;
}

void backtracking(int i, int j, vector<vector<char>> &board, string &word,
    vector<vector<bool>> &visited, bool &find, int wordPos) {
    if (find || board[i][j] != word[wordPos])
        return;
    
    if (wordPos == word.size() - 1) {
        find = true;
        return;
    }

    visited[i][j] = true;
    for (int k = 0; k < 4; ++k) {
        int row = i + directions[k];
        int col = j + directions[k+1];
        if (row >= 0 && row < board.size() && col >= 0 && col < board[0].size() && 
            !visited[row][col]) {
            backtracking(row, col, board, word, visited, find, wordPos + 1);
        }
    }
    visited[i][j] = false;

}