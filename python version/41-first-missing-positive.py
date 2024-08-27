class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            if nums[i] <= 0:
                nums[i] = n+1
        for i in range(n):
            original_num = abs(nums[i])
            if original_num <= n:
                if nums[original_num-1] > 0:
                    nums[original_num-1] = -nums[original_num-1]
                    
        for i in range(n):
            if nums[i] > 0:
                return i+1
        
        return n+1