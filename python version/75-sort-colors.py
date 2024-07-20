class Solution:
    def sortColors(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        zero = 0
        two = len(nums)-1
        curr = 0
        while curr <= two:
            if nums[curr] == 0:
                nums[curr], nums[zero] = nums[zero], nums[curr]
                curr += 1
                zero += 1
            elif nums[curr] == 2:
                nums[curr], nums[two] = nums[two], nums[curr]
                two -= 1
            else:
                curr += 1
        