class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        max_value = -10000-1
        n = len(nums)
        pre_sum = [0 for _ in range(n+1)]
        for i in range(1, n+1):
            pre_sum[i] = pre_sum[i-1] + nums[i-1]

        for i in range(0, n-k+1):
            curr = pre_sum[i+k]-pre_sum[i]
            max_value = max(max_value, curr / k)
        return max_value