class Solution:
    def solve(self, board: List[List[str]]) -> None:
        """
        Do not return anything, modify board in-place instead.
        """
        m = len(board)
        n = len(board[0])
        directions = [-1, 0, 1, 0, -1]
        visited = [[False] * n for _ in range(m)]

        def dfs(i, j):
            if i<0 or i>=m or j<0 or j>=n: return
            if board[i][j] == 'X' or visited[i][j]: return
            visited[i][j] = True
            for k in range(4):
                dfs(i+directions[k], j+directions[k+1])
        
        for j in range(n):
            if board[0][j] == 'O' and visited[0][j] == False:
                dfs(0, j)
            if board[m-1][j] == 'O' and visited[m-1][j] == False:
                dfs(m-1, j)

        for i in range(m):
            if board[i][0] == 'O' and visited[i][0] == False:
                dfs(i, 0)
            if board[i][n-1] == 'O' and visited[i][n-1] == False:
                dfs(i, n-1)


        for i in range(m):
            for j in range(n):
                if board[i][j] == 'O' and visited[i][j] == False:
                    board[i][j] = 'X'