// 2-dimension
int coinChange(vector<int>& coins, int amount) {
    int n = coins.size(), m = amount;
    vector<vector<int>> dp(n+1, vector<int>(m+1, INT_MAX-1));
    dp[0][0] = 0;
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j <= m; ++j) {
            if (j >= coins[i-1]) {
                dp[i][j] = min(dp[i-1][j], dp[i][j-coins[i-1]] + 1);
            }
            else {
                dp[i][j] = dp[i-1][j];
            }
        }
    }
    return dp[n][m] == INT_MAX-1 ? -1 : dp[n][m];
}

// 1-dimension
int coinChange(vector<int>& coins, int amount) {
    vector<int> dp(amount+1, INT_MAX-1);
    dp[0] = 0;
    for (const auto &coin : coins) {
        for (int i = 1; i <= amount; ++i) {
            if (i >= coin) {
                dp[i] = min(dp[i], dp[i-coin] + 1);
            }
        }
    }
    return dp[amount] == INT_MAX-1 ? -1 : dp[amount];
}