class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prev = [1 for _ in range(n)]
        after = [1 for _ in range(n)]
        for i in range(1, n):
            prev[i] = prev[i-1] * nums[i-1]
        
        for i in range(n-2, -1, -1):
            after[i] = after[i+1] * nums[i+1]
        
        ans = [x*y for x, y in zip(prev, after)]
        return ans