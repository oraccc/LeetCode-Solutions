class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        if n==1: return nums[0]
        dp = [0]*n
        dp[0] = nums[0]
        for i in range(1,n):
            dp[i] = max(dp[i-1] + nums[i], nums[i])
        
        return max(dp)