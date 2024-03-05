class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        curr = []
        ans = []
        n = len(candidates)

        def backtracking(start):
            if sum(curr) > target or start == n:
                return
            if sum(curr) == target:
                ans.append(curr[:])
                return
            for i in range(start, n):
                curr.append(candidates[i])
                backtracking(i)
                curr.pop()

        backtracking(0)

        return ans