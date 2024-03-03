class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False for _ in range(n)] for _ in range(n)]
        max_len = 1
        max_start = 0
        for i in range(n):
            if i < n-1 and s[i] == s[i+1]:
                dp[i][i+1] = True
                max_len = 2
                max_start = i
            dp[i][i] = True
        
        for s_len in range(3, n+1):
            for i in range(0, n-s_len+1):
                j = i+s_len-1
                if dp[i+1][j-1] and s[i] == s[j]:
                    dp[i][j] = True 
                    if s_len > max_len:
                        max_len = s_len 
                        max_start = i
        return s[max_start:max_start+max_len]
    

# Another way to dp

class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        dp = [[False for _ in range(n)] for _ in range(n)]
        max_len = 1
        max_start = 0
        
        for i in range(n-1, -1, -1):
            for j in range(i, n):
                if j-i <= 1 and s[j] == s[i]:
                    dp[i][j] = True
                    if j-i+1 > max_len:
                        max_len = j-i+1
                        max_start = i
                elif dp[i+1][j-1] and s[j] == s[i]:
                    dp[i][j] = True
                    if j-i+1 > max_len:
                        max_len = j-i+1
                        max_start = i

        return s[max_start:max_start+max_len]