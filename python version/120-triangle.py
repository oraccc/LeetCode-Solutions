class Solution:
    def minimumTotal(self, triangle: List[List[int]]) -> int:
        n = len(triangle)
        dp = [[0] * i for i in range(1, n+1)]
        dp[0][0] = triangle[0][0]
        for i in range(1, n):
            dp[i][0] = triangle[i][0] + dp[i-1][0]
            for j in range(1, i):
                dp[i][j] = triangle[i][j] + min(dp[i-1][j], dp[i-1][j-1])
            dp[i][i] = triangle[i][i] + dp[i-1][i-1]

        return min(dp[n-1])