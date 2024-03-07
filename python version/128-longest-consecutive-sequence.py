class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0 or n == 1:
            return n
        nums = set(nums)
        max_len = 1
        for num in nums:
            if num-1 not in nums:
                start = num
                while num+1 in nums:
                    num += 1
                max_len = max(num-start+1, max_len)
        return max_len