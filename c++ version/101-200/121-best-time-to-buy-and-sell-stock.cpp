int maxProfit(vector<int>& prices) {
    int minPrice = INT_MAX;
    int maxMoney = 0;
    for (int i = 0; i < prices.size(); ++i) {
        if (prices[i] < minPrice) {
            minPrice = prices[i];
        }
        else {
            maxMoney = max(maxMoney, prices[i] - minPrice);
        }
    }
    return maxMoney;
}