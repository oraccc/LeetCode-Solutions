vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> ans;
    if (n == 0) 
        return ans;
    vector<string> borad(n, string(n, '.'));
    vector<bool> column(n, false), ldiag(2*n-1, false), rdiag(2*n-1, false);
    backtracking(borad, column, ldiag, rdiag, ans, 0, n);
    return ans;
}

void backtracking(vector<string> &borad, vector<bool> &column, vector<bool> &ldiag,
    vector<bool> &rdiag, vector<vector<string>> &ans, int row, int n) {
    if (row == n) {
        ans.push_back(borad);
        return;
    }
    for (int col = 0; col < n; ++col) {
        if (column[col] || ldiag[n-row-1+col] || rdiag[col+row])
            continue;
        borad[row][col] = 'Q';
        column[col] = ldiag[n-row-1+col] = rdiag[col+row] = true;
        backtracking(borad, column, ldiag, rdiag, ans, row + 1, n);
        column[col] = ldiag[n-row-1+col] = rdiag[col+row] = false;
        borad[row][col] = '.';
    }
}