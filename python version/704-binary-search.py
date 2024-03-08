class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums)
        while left < right:
            mid = left + (right-left)//2
            if nums[mid] >= target:
                right = mid
            else:
                left = mid + 1
        
        if left < len(nums) and nums[left] == target: return left
        else: return -1