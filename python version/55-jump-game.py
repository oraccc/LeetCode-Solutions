class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 1: return True
        dp = [False for i in range(n)]
        dp[0] = True
        for i in range(1, n):
            for k in range(1, i+1):
                if dp[i-k] and nums[i-k] >= k:
                    dp[i] = True
                    break
             
        return dp[n-1]
            