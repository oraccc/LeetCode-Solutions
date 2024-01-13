int countSubstrings(string s) {
    int n = s.size();
    vector<vector<bool>> dp(n, vector<bool>(n, false));

    int count = 0;
    for (int i = n-1; i >= 0; --i) {
        for (int j = i; j < n; ++j) {
            if (j - i <= 1) {
                if (s[i] == s[j]) {
                    ++count;
                    dp[i][j] = true;
                }
            }
            else if (dp[i+1][j-1] == true && s[i] == s[j]) {
                ++count;
                dp[i][j] = true;
            }
        }
    }
    return count;
}