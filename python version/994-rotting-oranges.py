class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        count = 0
        directions = [-1, 0, 1, 0, -1]
        queue = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    count += 1
                elif grid[i][j] == 2:
                    queue.append((i, j))
        ans = 0
        if count == 0:
            return ans
        while queue:
            ans += 1
            for _ in range(len(queue)):
                i, j = queue[0]
                queue.pop(0)
                for k in range(4):
                    row = i + directions[k]
                    col = j + directions[k+1]
                    if row < m and row >= 0 and col < n and col >=0 and grid[row][col]==1:
                        count -= 1
                        grid[row][col] = 2
                        queue.append((row, col))
        if count != 0:
            return -1
        else:
            return ans-1
