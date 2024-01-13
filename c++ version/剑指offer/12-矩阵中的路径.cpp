class Solution {
    vector<int> directions{-1, 0, 1, 0, -1};
public:
    bool exist(vector<vector<char>>& board, string word) {
        int m = board.size(), n = board[0].size();
        vector<vector<bool>> visited(m, vector<bool>(n, false));
        bool found = false;
        int currPos = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                backtracking(board, word, i, j, currPos, found, visited);
                if (found) return found;
            }
        }

        return found;
    }

    void backtracking(vector<vector<char>>& board, string& word, int i, int j, int currPos, bool &found, 
        vector<vector<bool>> &visited) {
        if (found || board[i][j] != word[currPos] || visited[i][j]) return;

        if (currPos == word.size()-1) {
            found = true;
            return;
        }
        visited[i][j] = true;
        for (int k = 0; k < 4; ++k) {
            int row = i + directions[k];
            int col = j + directions[k+1];
            if (row >= 0 && row < board.size() && col >= 0 && col < board[0].size()) {
                backtracking(board, word, row, col, currPos+1, found, visited);
            }
        }
        visited[i][j] = false;
    }
};