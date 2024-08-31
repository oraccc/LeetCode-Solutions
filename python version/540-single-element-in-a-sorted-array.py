class Solution:
    def singleNonDuplicate(self, nums: List[int]) -> int:
        if len(nums) == 1:
            return nums[0]
        left = 0
        right = len(nums)-1

        while left < right:
            mid = (left+right)//2
            if mid % 2 == 1:
                if nums[mid] == nums[mid-1]:
                    left = mid+1
                else:
                    right = mid 
            else:
                if nums[mid] == nums[mid+1]:
                    left = mid+1
                else:
                    right = mid
        return nums[left]