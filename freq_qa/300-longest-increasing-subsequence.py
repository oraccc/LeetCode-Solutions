class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        max_length = 1
        n = len(nums)
        dp = [1]*n
        for i in range(n):
            for j in range(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j]+1)
            
            max_length = max(max_length, dp[i])

        return max_length