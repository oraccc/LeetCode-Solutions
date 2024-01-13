int rob(vector<int>& nums) {
    int tmp = 0;
    int n = nums.size();
    if (n == 1) return nums[0];
    // rob from 1 to n-1
    vector<int> dp(n, 0);
    dp[1] = nums[0];
    for (int i = 2; i <= n-1; ++i) {
        dp[i] = max(dp[i-1], dp[i-2] + nums[i-1]);
    }
    tmp = dp[n-1];
    // rob from 2 to n
    dp = vector<int>(n, 0);
    dp[1] = nums[1];
    for (int i = 3; i <= n; ++i) {
        dp[i-1] = max(dp[i-2], dp[i-3] + nums[i-1]);
    }

    return max(tmp, dp[n-1]);
}