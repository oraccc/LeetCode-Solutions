int numWays(int n) {
    int MOD = 1000000007;
    if (n == 0) return 1;
    if (n <= 2) return n;
    vector<int> dp(n+1, 1);
    for (int i = 2; i <= n; ++i) {
        dp[i] = (dp[i-1] + dp[i-2]) % MOD;
    }
    return dp[n] % MOD;
}