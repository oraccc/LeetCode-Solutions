class Solution:
    def partition(self, s: str) -> List[List[str]]:
        n = len(s)

        ans = []
        curr = []

        def backtracking(start):
            if start == n:
                ans.append(curr[:])
                return
            for end in range(start+1, n+1):
                substr = s[start:end]
                if is_valid(substr):
                    curr.append(substr)
                    backtracking(end)
                    curr.pop()
        
        def is_valid(substr):
            left = 0
            right = len(substr)-1
            while left < right:
                if substr[left] != substr[right]:
                    return False
                else:
                    left += 1
                    right -= 1
            return True

        backtracking(0)

        return ans