// 3-dimensions
int findMaxForm(vector<string>& strs, int m, int n) {
    int s = strs.size();
    vector<vector<vector<int>>> dp(s+1, vector<vector<int>>(m+1, vector<int>(n+1, 0)));
    for(int k = 1; k <= s; ++k) {
        auto countPair = count01(strs[k-1]);
        int count0 = countPair.first, count1 = countPair.second;
        for (int i = 0; i <= m; ++i) {
            for (int j = 0; j <= n; ++j) {
                if (i >= count0 && j >= count1) {
                    dp[k][i][j] = max(dp[k-1][i][j], 1 + dp[k-1][i-count0][j-count1]);
                }
                else {
                    dp[k][i][j] = dp[k-1][i][j];
                }
            }
        }
    }
    return dp[s][m][n];
}

// 2-dimensions (reverse walk through)
int findMaxForm(vector<string>& strs, int m, int n) {
    int s = strs.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for(const string &str : strs) {
        auto countPair = count01(str);
        int count0 = countPair.first, count1 = countPair.second;
        for (int i = m; i >= count0; --i) {
            for (int j = n; j >= count1; --j) {
                dp[i][j] = max(dp[i][j], 1 + dp[i-count0][j-count1]);
            }
        }
    }
    return dp[m][n];
}

pair<int, int> count01(const string &s) {
    int count0 = s.size(), count1 = 0;
    for (const auto &c : s) {
        if (c == '1') {
            ++count1;
            --count0;
        }
    }
    return make_pair(count0, count1);
}