class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        comb = []
        ans = []

        def backtracking(start):
            if len(comb)+n-start+1<k:
                return
            if len(comb) == k:
                ans.append(comb[:])
                return
            for i in range(start, n+1):
                comb.append(i)
                backtracking(i+1)
                comb.pop()
        
        backtracking(1)
        return ans
            