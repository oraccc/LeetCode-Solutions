bool isValidSudoku(vector<vector<char>>& board) {
    vector<vector<bool>> rowRecord(9, vector<bool>(9, false));
    vector<vector<bool>> colRecord(9, vector<bool>(9, false));
    vector<vector<bool>> gridRecord(9, vector<bool>(9, false));

    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (board[i][j] != '.') {
                int index = int (board[i][j]-'0') - 1;
                int gridId = 3*(i/3) + j/3;
                if (!rowRecord[i][index] && !colRecord[j][index] && !gridRecord[gridId][index])
                    rowRecord[i][index] = colRecord[j][index] = gridRecord[gridId][index] = true;
                else return false;
            }
        }
    }

    return true;
}