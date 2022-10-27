vector<vector<int>> combine(int n, int k) {
    vector<vector<int>> ans;
    vector<int> comb;
    backtracking(1, n, k, ans, comb);
    return ans;
}

void backtracking(int i, int n, int k, vector<vector<int>> &ans, vector<int> &comb) {
    if (comb.size() == k) {
        ans.push_back(comb);
        return;
    }
    for (int j = i; j <= n; ++j) {
        comb.push_back(j);
        backtracking(j+1, n, k, ans, comb);
        comb.pop_back();
    }
}