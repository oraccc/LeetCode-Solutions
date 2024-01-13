//2-dimension
int minDistance(string word1, string word2) {
    int n = word1.size(), m = word2.size();
    vector<vector<int>> dp(n+1, vector<int>(m+1, INT_MAX-1));
    dp[0][0] = 0;
    for (int i = 1; i <= n; ++i) {
        dp[i][0] = i;
    }
    for (int j = 1; j <= m; ++j) {
        dp[0][j] = j;
    }

    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (word1[i-1] == word2[j-1]) {
                dp[i][j] = dp[i-1][j-1];
            }
            else {
                dp[i][j] = min(dp[i-1][j-1] + 1, min(dp[i][j-1] + 1, dp[i-1][j] + 1));
            }
        }
    }
    return dp[n][m];
}

//1-dimension hard to understand through code only
int minDistance(string word1, string word2) {
    int n = word1.size(), m = word2.size();
    vector<int> dp(m+1, INT_MAX-1);
    for (int j = 0; j <= m; ++j) {
        dp[j] = j;
    }

    for (int i = 1; i <= n; ++i) {
        int pre1 = dp[0];
        dp[0] = i;
        for (int j = 1; j <= m; ++j) {
            int pre2 = dp[j];
            if (word1[i-1] == word2[j-1]) {
                dp[j] = pre1;
            }
            else {
                dp[j] = min(pre1 + 1, min(pre2 + 1, dp[j-1] + 1));
            }
            pre1 = pre2;
        }
    }
    return dp[m];
}