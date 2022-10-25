vector<vector<int>> permuteUnique(vector<int>& nums) {
    vector<vector<int>> ans;
    backtracking(nums, 0, ans);
    return ans;
}

void backtracking(vector<int> &nums, int pos, vector<vector<int>> &ans) {
    if (pos == nums.size() - 1) {
        ans.push_back(nums);
        return;
    }
    unordered_set<int> record;
    for (int i = pos; i < nums.size(); ++i) {
        if (record.insert(nums[i]).second) {
            swap(nums[i], nums[pos]);
            backtracking(nums, pos + 1, ans);
            swap(nums[i], nums[pos]);   
        }
        
    }
}