class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        cols = [False] * n
        l_diags = [False] * (2*n-1)
        r_diags = [False] * (2*n-1)
        board = [["."]*n for _ in range(n)]
        ans = []
        def dfs(row):
            if row == n:
                tmp = ["".join(i) for i in board]
                ans.append(tmp)
                return
            for col in range(n):
                if cols[col] or l_diags[row+col] or r_diags[n-1-col+row]: continue
                cols[col] = l_diags[row+col] = r_diags[n-1-col+row] = True
                board[row][col] = "Q"
                dfs(row+1)
                board[row][col] = "."
                cols[col] = l_diags[row+col] = r_diags[n-1-col+row] = False
        dfs(0)
        return ans

