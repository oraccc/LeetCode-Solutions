vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
    vector<int> path;
    vector<vector<int>> ans;
    int sum = 0, startPos = 0;
    sort(candidates.begin(), candidates.end());
    backtracking(startPos, sum, candidates, target, path, ans);
    return ans;
}

void backtracking(int startPos, int sum, vector<int> &candidates, int target, vector<int> &path, vector<vector<int>> &ans) {
    if (sum == target) {
        ans.push_back(path);
        return;
    }

    for (int i = startPos; i < candidates.size(); ++i) {
        if (sum + candidates[i] > target) break;
        path.push_back(candidates[i]);
        sum += candidates[i];
        backtracking(i, sum, candidates, target, path, ans);
        sum -= candidates[i];
        path.pop_back();
    }
}