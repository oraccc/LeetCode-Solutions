class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [10001 for i in range(amount+1)]
        dp[0] = 0
        for i in range(1, amount+1):
            for j in coins:
                if i >= j:
                    dp[i] = min(dp[i], dp[i-j]+1)
        
        if dp[amount] == 10001: return -1
        else: return dp[amount]