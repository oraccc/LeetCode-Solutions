vector<vector<int>> permute(vector<int>& nums) {
    vector<vector<int>> ans;
    backtracking(nums, 0, ans);
    return ans;
}

void backtracking(vector<int> &nums, int pos, vector<vector<int>> &ans) {
    if (pos == nums.size() - 1) {
        ans.push_back(nums);
        return;
    }
    for (int i = pos; i < nums.size(); ++i) {
        swap(nums[i], nums[pos]);
        backtracking(nums, pos + 1, ans);
        swap(nums[i], nums[pos]);
    }
}