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

// Another solution: check before push answers (too much time cost)

vector<vector<int>> permuteUnique(vector<int>& nums) {
    vector<vector<int>> ans;
    vector<int> currPerm;
    backtracking(nums, 0, currPerm, ans);
    return ans;
}

void backtracking(vector<int> &nums, int pos, vector<int> &currPerm, 
    vector<vector<int>> &ans) {
    if (pos == nums.size()) {
        if (find(ans.begin(), ans.end(), currPerm) == ans.end())
            ans.push_back(currPerm);
        return;
    }
    for (int i = pos; i < nums.size(); ++i) {
        swap(nums[i], nums[pos]);
        currPerm.push_back(nums[pos]);
        backtracking(nums, pos + 1, currPerm, ans);
        currPerm.pop_back();
        swap(nums[i], nums[pos]);   
    }
}