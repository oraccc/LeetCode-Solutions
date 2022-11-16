int findTargetSumWays(vector<int>& nums, int target) {
    int currSum = 0, count = 0;
    dfs(0, currSum, nums, target, count);
    return count;
}

void dfs(int index, int currSum, vector<int> &nums, int target, int &count) {
    if (index == nums.size()) {
        if (currSum == target)
            ++count;
        return;
    }
    dfs(index+1, currSum + nums[index], nums, target, count);
    dfs(index+1, currSum -= nums[index], nums, target, count);
}