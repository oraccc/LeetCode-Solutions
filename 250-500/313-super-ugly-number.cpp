int nthSuperUglyNumber(int n, vector<int>& primes) {
    vector<long long> dp(n, INT_MAX);
    dp[0] = 1;
    vector<int> pointers(primes.size(), 0);
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < primes.size(); ++j) {
            dp[i] = min(dp[i], primes[j] * dp[pointers[j]]);
        }
        for (int j = 0; j < primes.size(); ++j) {
            if (dp[i] == dp[pointers[j]] * primes[j]) {
                ++pointers[j];
            }
        }
    }
    return dp[n-1];
}