int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    if (n == 1) return 1;
    vector<int> dp(n, 1);
    int maxLength = 0;
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (nums[j] < nums[i]) {
                dp[i] = max(dp[j] + 1, dp[i]);
            }
        }
        maxLength = max(maxLength, dp[i]);
    }

    return maxLength;
}