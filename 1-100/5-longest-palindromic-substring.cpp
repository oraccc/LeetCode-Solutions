string longestPalindrome(string s) {
    int n = s.size();
    if (n == 1) return s;
    int maxLength = 1;
    int maxStart = 0;
    vector<vector<bool>> dp(n, vector<bool>(n, false));

    for (int i = 0; i < n; ++i) {
        dp[i][i] = true;
        if (i < n - 1) {
            if (s[i] == s[i+1]) {
                dp[i][i+1] = true;
                maxLength = 2;
                maxStart = i;
            }
        }
    }

    for (int len = 3; len <= n; ++len) {
        for (int i = 0; i + len - 1 < n; ++i) {
            int j = i + len - 1;
            if (s[j] == s[i] && dp[i+1][j-1] == true) {
                dp[i][j] = true;
                maxStart = i;
                maxLength = len;
            }
        }
    }

    return s.substr(maxStart, maxLength);
}