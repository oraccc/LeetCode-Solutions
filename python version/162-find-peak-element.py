class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        left, right = 0, len(nums)
        while left < right:
            mid = left + (right-left)//2
            if mid == len(nums)-1: right = mid
            elif nums[mid] < nums[mid+1]:
                left = mid+1
            else:
                right = mid
        return left