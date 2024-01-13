int maxCoins(vector<int>& nums) {
    int n = nums.size();
    vector<int> ballons(n+2, 1);
    for (int i = 1; i < n+1; ++i) {
        ballons[i] = nums[i-1];
    }

    vector<vector<int>> dp(n+2, vector<int>(n+2, 0));
    for (int i = n; i >= 0; --i) {
        for (int j = i+1; j <= n+1; ++j) {
            for (int k = i+1; k <= j-1; ++k) {
                dp[i][j] = max(dp[i][j], dp[i][k] + dp[k][j] + 
                    ballons[i] * ballons[k] * ballons[j]);
            }
        }
    }
    return dp[0][n+1];
}