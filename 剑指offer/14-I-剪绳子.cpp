class Solution {
public:
    int cuttingRope(int n) {
        vector<int> dp(n+1, 0);
        dp[2] = 1;
        for (int i = 3; i <= n; ++i) {
            for (int j = 1; j < n; ++j) {
                dp[i] = max(dp[i], max(j * (i-j), dp[j] * (i-j)));
            }
        }

        return dp[n];
    }
};