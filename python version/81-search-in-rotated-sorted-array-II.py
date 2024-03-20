class Solution:
    def search(self, nums: List[int], target: int) -> bool:
        left, right = 0, len(nums)-1
        while left < right:
            mid = left + (right-left)//2
            if nums[mid] == target:
                return True
            if nums[left] == nums[right]:
                left += 1
                continue
            if nums[mid] <= nums[right]:
                if nums[mid] < target and nums[right] >= target:
                    left = mid+1
                else:
                    right = mid-1
            else:
                if nums[mid] > target and nums[left] <= target:
                    right = mid-1
                else:
                    left = mid+1

        if nums[left] == target: return True
        else: return False