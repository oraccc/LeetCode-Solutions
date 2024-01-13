int maxProfit(vector<int>& prices) {
    int n = prices.size();
    if (n == 1) return 0;
    vector<int> buy1(n+1, 0), sell1(n+1, 0), buy2(n+1, 0), sell2(n+1, 0);
    buy1[0] = -prices[0];
    buy2[0] = -prices[0];
    for (int i = 1; i <= n; ++i) {
        buy1[i] = max(buy1[i-1], -prices[i-1]);
        sell1[i] = max(buy1[i-1] + prices[i-1], sell1[i-1]);
        buy2[i] = max(buy2[i-1], sell1[i-1] - prices[i-1]);
        sell2[i] = max(buy2[i-1] + prices[i-1], sell2[i-1]);
    }

    return sell2[n];
}