int numDecodings(string s) {
    int n = s.size();
    int prev = s[0] - '0';
    if (!prev) return 0;
    if (n == 1) return 1;
    vector<int> dp(n+1, 1);
    for (int i = 2; i <= n; ++i) {
        int curr = s[i-1] - '0';
        if ((prev == 0 || prev > 2) && curr == 0) 
            return 0;
        else if (prev == 1 || (prev == 2 && curr <= 6)) {
            if (curr != 0) {
                dp[i] = dp[i-1] + dp[i-2];
            }
            else {
                dp[i] = dp[i-2];
            }
        }
        else {
            dp[i] = dp[i-1];
        }
        prev = curr;
    }

    return dp[n];
}