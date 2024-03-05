class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        dp = [0] * n
        for i in range(m):
            if i == 0:
                dp[0] = grid[0][0]
                for j in range(1, n):
                    dp[j] = dp[j-1] + grid[0][j]
            else:
                dp[0] = dp[0] + grid[i][0]
                for j in range(1, n):
                    dp[j] = min(dp[j], dp[j-1]) + grid[i][j]

        return dp[n-1]