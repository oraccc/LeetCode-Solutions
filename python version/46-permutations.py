class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        visited = [False for _ in range(n)]
        curr = []
        ans = []

        def backtracking():
            if len(curr) == n:
                ans.append(curr[:])
                return
            for i in range(n):
                if visited[i] == False:
                    visited[i] = True
                    curr.append(nums[i])
                    backtracking()
                    curr.pop()
                    visited[i] = False
        
        backtracking()
        return ans
