int maxProfit(vector<int>& prices, int fee) {
    int n = prices.size();
    if (n == 1) return 0;
    vector<int> buy(n+1, 0), sell(n+1, 0);
    buy[0] = -prices[0];

    for (int i = 1; i <= n; ++i) {
        buy[i] = max(buy[i-1], sell[i-1] - prices[i-1]);
        sell[i] = max(sell[i-1], buy[i-1] + prices[i-1] - fee);
    }

    return sell[n];
}