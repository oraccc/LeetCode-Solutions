class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        n = len(nums)
        min_length = 100001
        left = 0
        curr_sum = 0
        for right in range(n):
            curr_sum += nums[right]
            while curr_sum >= target:
                size = right - left + 1
                min_length = min(size, min_length)
                curr_sum -= nums[left]
                left += 1
        
        if min_length == 100001: return 0
        return min_length