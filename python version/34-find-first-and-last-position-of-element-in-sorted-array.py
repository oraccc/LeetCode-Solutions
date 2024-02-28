class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        lower = self.find_lower_bound(nums, target)
        upper = self.find_upper_bound(nums, target)
        if lower == len(nums) or nums[lower] != target:
            return [-1, -1]
        return [lower, upper]
        

    def find_lower_bound(self, nums, target):
        left = 0
        right = len(nums)
        while(left < right):
            mid = left + (right-left)//2
            if nums[mid] >= target:
                right = mid
            else:
                left = mid+1
            
        return left

    def find_upper_bound(self, nums, target):
        left = 0
        right = len(nums)
        while(left < right):
            mid = left + (right-left)//2
            if nums[mid] > target:
                right = mid
            else:
                left = mid+1
            
        return left-1