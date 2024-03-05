class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        count = 0
        directions = [-1, 0, 1, 0, -1]
        m = len(grid)
        n = len(grid[0])
        # visited = [[False] * n for _ in range(m)]

        def dfs(i, j):
            if i<0 or i>=m or j <0 or j>=n: return
            if grid[i][j] == "0": return
            
            grid[i][j] = "0"
            for k in range(4):
                dfs(i+directions[k], j+directions[k+1])

        for i in range(m):
            for j in range(n):
                if grid[i][j]=="1":
                    count+=1
                    dfs(i,j)

        return count



        