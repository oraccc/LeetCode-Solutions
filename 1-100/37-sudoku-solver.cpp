void solveSudoku(vector<vector<char>>& board) {
    vector<vector<bool>> rowRecord(9, vector<bool>(9, false));
    vector<vector<bool>> colRecord(9, vector<bool>(9, false));
    vector<vector<bool>> gridRecord(9, vector<bool>(9, false));

    vector<vector<char>> answer(9, vector<char>(9, '.'));
    for (int i = 0; i < 9; ++i) {
        for (int j = 0; j < 9; ++j) {
            if (board[i][j] != '.') {
                answer[i][j] = board[i][j];
                int index = int (board[i][j]-'0') - 1;
                int gridId = 3*(i/3) + j/3;
                rowRecord[i][index] = true;
                colRecord[j][index] = true;
                gridRecord[gridId][index] = true;
            }
        }
    }
    int k = 0;
    backtracking(k, answer, board, rowRecord, colRecord, gridRecord);
}

void backtracking(int k, vector<vector<char>> &answer, vector<vector<char>> &board, 
    vector<vector<bool>> &rowRecord,vector<vector<bool>> &colRecord, vector<vector<bool>> &gridRecord) {
    if (k == 81) {
        for (int x = 0; x < 9; ++x) {
            for (int y = 0; y < 9; ++y) {
                board[x][y] = answer[x][y];
            }
        }
        return;
    }
    int i = k/9, j = k%9;
    if (answer[i][j] != '.') {
        backtracking(k+1, answer, board, rowRecord, colRecord, gridRecord);
        return;
    }
    for (int num = 1; num <= 9; ++num) {
        int gridId = 3*(i/3) + j/3;
        if (rowRecord[i][num-1] || colRecord[j][num-1] || gridRecord[gridId][num-1])
            continue;
        answer[i][j] = num + '0';
        rowRecord[i][num-1] = colRecord[j][num-1] = gridRecord[gridId][num-1] = true;
        backtracking(k+1, answer, board, rowRecord, colRecord, gridRecord);
        answer[i][j] = '.';
        rowRecord[i][num-1] = colRecord[j][num-1] = gridRecord[gridId][num-1] = false;
    }
}