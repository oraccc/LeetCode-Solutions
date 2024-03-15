class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1
        while left < right:
            mid = left + (right-left)//2
            if nums[mid] == target:
                return mid
            if nums[mid] < nums[right]:
                if nums[mid] < target and nums[right] >= target:
                    left = mid+1
                else:
                    right = mid
            else:
                if nums[mid] > target and nums[left] <= target:
                    right = mid
                else:
                    left = mid+1

        if nums[left] == target: return left
        else: return -1