class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        n = len(nums)
        if n <= 1:
            return
        
        i = n-2
        j = n-1
        while i >= 0 and nums[i] >= nums[j]:
            i -= 1
            j -= 1
        
        if i == -1:
            left = 0
            right = n-1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
            return
        else:
            k = n-1
            while k > i and nums[i] >= nums[k]:
                k -= 1
            nums[i], nums[k] = nums[k], nums[i]
            left = j
            right = n-1
            while left < right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
            return