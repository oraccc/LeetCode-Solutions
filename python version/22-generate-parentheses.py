class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans = []
        s = []
        self.left = 0
        self.right = 0

        def dfs():
            if len(s) == 2*n:
                if self.left == self.right:
                    tmp = "".join(s)
                    ans.append(tmp)
                return
            if self.right > self.left : return
            s.append("(")
            self.left += 1
            dfs()
            self.left -= 1
            s.pop()
            s.append(")")
            self.right += 1
            dfs()
            self.right -= 1
            s.pop()
        dfs()
        return ans