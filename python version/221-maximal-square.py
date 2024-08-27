class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        n = len(matrix)
        m = len(matrix[0])

        dp = [[0]*m for _ in range(n)]
        max_len = 0
        for i in range(n):
            if matrix[i][0] == "1":
                dp[i][0] = 1
                max_len = 1
        
        for j in range(m):
            if matrix[0][j] == "1":
                dp[0][j] = 1
                max_len = 1
        
        for i in range(1,n):
            for j in range(1,m):
                if matrix[i][j] == "1":
                    dp[i][j] = min(dp[i-1][j], dp[i-1][j-1], dp[i][j-1])+1
                    max_len = max(dp[i][j], max_len)

        return max_len * max_len