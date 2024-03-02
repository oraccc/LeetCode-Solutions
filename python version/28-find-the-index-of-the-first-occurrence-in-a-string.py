class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        n = len(haystack)
        m = len(needle)
        for i in range(n):
            start = i
            j = 0
            while start < n and j < m:
                if needle[j] == haystack[start]:
                    start += 1
                    j += 1
                else:
                    break
            if j == m:
                return start - m
        return -1
