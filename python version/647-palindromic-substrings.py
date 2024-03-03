class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False for _ in range(n)] for _ in range(n)]
        count = 0
        for i in range(n):
            if i < n-1 and s[i] == s[i+1]:
                dp[i][i+1] = True
                count += 1
            dp[i][i] = True
            count += 1
        
        for s_len in range(3, n+1):
            for i in range(0, n-s_len+1):
                j = i + s_len - 1
                if s[i] == s[j] and dp[i+1][j-1]:
                    dp[i][j] = True
                    count += 1
                
        return count

# Another way to dp
class Solution:
    def countSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [[False for _ in range(n)] for _ in range(n)]
        count = 0
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if j-i <= 1 and s[j] == s[i]:
                    dp[i][j] = True
                    count += 1
                elif dp[i+1][j-1] and s[j] == s[i]:
                    dp[i][j] = True
                    count += 1
                
        return count