class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        curr = []
        ans.append(curr[:])
        def backtracking(i):
            for j in range(i, n):
                curr.append(nums[j])
                ans.append(curr[:])
                backtracking(j+1)
                curr.pop()
        backtracking(0)
        return ans