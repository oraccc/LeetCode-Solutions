vector<vector<int>> combinationSum2(vector<int>& candidates, int target) {
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

    unordered_set<int> record;
    for (int i = startPos; i < candidates.size(); ++i) {
        if (sum + candidates[i] > target) break;
        if (record.insert(candidates[i]).second) {
            path.push_back(candidates[i]);
            sum += candidates[i];
            backtracking(i+1, sum, candidates, target, path, ans);
            sum -= candidates[i];
            path.pop_back();
        }
    }
}