class Solution:
    def maximumProduct(self, nums: List[int]) -> int:
        nums.sort()
        if nums[0] >= 0:
            return nums[-1]*nums[-2]*nums[-3]
        elif nums[-1] <= 0:
            return nums[-1]*nums[-2]*nums[-3]
        else:
            max_value = max(nums[-1]*nums[-2]*nums[-3], nums[0]*nums[1]*nums[-1])
            return max_value