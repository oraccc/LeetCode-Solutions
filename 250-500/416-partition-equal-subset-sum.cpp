bool canPartition(vector<int>& nums) {
    int sum = accumulate(nums.begin(), nums.end(), 0);
    if (sum % 2) return false;
    int n = nums.size();
    int target = sum / 2;
    vector<vector<int>> dp(n+1, vector<int>(target+1, false));
    dp[0][0] = true;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= target; ++j) {
            if (j < nums[i-1]) {
                dp[i][j] = dp[i-1][j];
            }
            else {
                dp[i][j] = dp[i-1][j] || dp[i-1][j-nums[i-1]];
            }
        }
    }
    return dp[n][target];
}