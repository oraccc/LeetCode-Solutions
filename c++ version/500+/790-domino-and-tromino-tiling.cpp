int numTilings(int n) {
    long long mod = 1000000007;
    vector<vector<long long>> dp(n+1, vector<long long>(4, 0));
    dp[0][3] = 1;
    for(int i = 1; i <= n; ++i) {
        dp[i][0] = (dp[i - 1][3]) % mod;
        dp[i][1] = (dp[i - 1][2] + dp[i - 1][0]) % mod;
        dp[i][2] = (dp[i - 1][1] + dp[i - 1][0]) % mod;
        dp[i][3] = (dp[i - 1][0] + dp[i - 1][1] + dp[i - 1][2] + dp[i - 1][3]) % mod;
    }
    return dp[n][3];
}