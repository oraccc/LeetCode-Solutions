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

// priority_queue: TLE for one case
int nthSuperUglyNumber(int n, vector<int>& primes) {
    unordered_set<long long> s;
    priority_queue<long long, vector<long long>, greater<long long>> q;
    s.insert(1);
    q.push(1);
    int head = 0;
    for (int i = 0; i < n; ++i) {
        head = q.top();
        for (const auto &prime : primes) {
            long long result = head * (long long)prime;
            if (s.count(result) == 0) {
                s.insert(result);
                q.push(result);
            }
        }
        q.pop();
    }
    return head;
}