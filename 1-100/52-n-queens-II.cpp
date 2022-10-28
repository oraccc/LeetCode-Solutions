int totalNQueens(int n) {
    int count = 0;
    if (n == 0) 
        return count;
    vector<bool> column(n, false), ldiag(2*n-1, false), rdiag(2*n-1, false);
    backtracking(column, ldiag, rdiag, count, 0, n);
    return count;
}

void backtracking(vector<bool> &column, vector<bool> &ldiag,
    vector<bool> &rdiag, int &count, int row, int n) {
    if (row == n) {
        ++count;
        return;
    }
    for (int col = 0; col < n; ++col) {
        if (column[col] || ldiag[n-row-1+col] || rdiag[col+row])
            continue;
        column[col] = ldiag[n-row-1+col] = rdiag[col+row] = true;
        backtracking(column, ldiag, rdiag, count, row + 1, n);
        column[col] = ldiag[n-row-1+col] = rdiag[col+row] = false;
    }
}