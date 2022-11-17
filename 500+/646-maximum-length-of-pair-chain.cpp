int findLongestChain(vector<vector<int>>& pairs) {
    int n = pairs.size();
    if (n == 1) return 1;
    vector<int> dp(n, 1);
    sort(pairs.begin(), pairs.end(), 
        [](const auto &pair1, const auto &pair2) {
            return pair1[0] < pair2[0];
        });

    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (pairs[i][0] > pairs[j][1]) {
                dp[i] = max(dp[i], dp[j] + 1);
            }
        }
    }

    return *max_element(dp.begin(), dp.end());
}