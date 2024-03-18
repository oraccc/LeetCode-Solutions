class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        n = len(nums)
        target = sum(nums)
        if target % 2 == 1:
            return False
        target = target // 2
        dp = [[False for _ in range(target+1)] for _ in range(n)]
        dp[0][0] = True
        if nums[0] <= target:
            dp[0][nums[0]] = True
        for i in range(1, n):
            for j in range(target+1):
                if nums[i] <= j:
                    dp[i][j] = dp[i-1][j] or dp[i-1][j-nums[i]]

                else:
                    dp[i][j] = dp[i-1][j]
        return dp[n-1][target]