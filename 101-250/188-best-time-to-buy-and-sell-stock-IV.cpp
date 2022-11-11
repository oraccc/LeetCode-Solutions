int maxProfit(int k, vector<int>& prices) {
    int n = prices.size();
    if (n == 1) return 0;
    if (k >= n/2) {
        int maxMoney = 0;
        for (int i = 1; i < n; ++i) {
            if (prices[i] > prices[i-1]) {
                maxMoney += (prices[i] - prices[i-1]);
            }
        }
        return maxMoney;
    }

    vector<vector<int>> hold(n+1, vector<int>(k+1, INT_MIN));
    vector<vector<int>> sold(n+1, vector<int>(k+1, 0));

    hold[0][0] = -prices[0];
    sold[0][0] = 0;

    for (int i = 1; i <= n; ++i) {
        hold[i][0] = max(hold[i-1][0], sold[i-1][0] - prices[i-1]);
    }
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= k; ++j) {
            hold[i][j] = max(hold[i-1][j], sold[i-1][j] - prices[i-1]);
            sold[i][j] = max(sold[i-1][j], hold[i-1][j-1] + prices[i-1]);
        }
    }

    return sold[n][k];
}