class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        direction = [-1, 0, 1, 0, -1]
        n = len(board)
        m = len(board[0])
        self.visited = [[False for _ in range(m)] for _ in range(n)]
        self.found = False

        def backtracking(i, j, curr):
            if board[i][j] != word[curr] or self.found: return
            if curr == len(word)-1:
                self.found = True 
                return
            self.visited[i][j] = True
            for k in range(4):
                row = i + direction[k]
                col = j + direction[k+1]
                if row >= 0 and row <= n-1 and col >= 0 and col <= m-1 and not self.visited[row][col]:
                    backtracking(row, col, curr+1)
            self.visited[i][j] = False

        for i in range(n):
            for j in range(m):
                if not self.found:
                    backtracking(i, j, 0)
        
        return self.found
