bool wordBreak(string s, vector<string>& wordDict) {
    int n = s.size();
    vector<bool> dp(n+1, false);
    dp[0] = true;
    for (int i = 1; i <= n; ++i) {
        for (const auto &word : wordDict) {
            int len = word.size();
            if (i >= len) {
                if (s.substr(i-len, len) == word) {
                    if (dp[i-len] == true) {
                        dp[i] = true;
                        break;
                    }
                }
            }
        }
    }
    return dp[n];
}