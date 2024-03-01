# DP
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1: return 0
        dp = [10001 for i in range(n)]
        dp[0] = 0
        for i in range(1, n):
            for k in range(1, i+1):
                if dp[i-k] != 10001 and nums[i-k] >= k:
                    dp[i] = min(dp[i], dp[i-k]+1)
        
        return dp[n-1]


# Greedy
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        max_distance = 0
        min_jump = 0
        start = 0
        while max_distance < n-1:
            min_jump += 1
            curr_distance = max_distance
            while start <= curr_distance:
                max_distance = max(max_distance, nums[start] + start)
                start += 1

        return min_jump