class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        prev = prices[0]
        for i in range(1, len(prices)):
            if prices[i] < prev:
                prev = prices[i]
            else:
                profit = max(profit, prices[i]-prev)

        return profit