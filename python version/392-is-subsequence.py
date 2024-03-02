class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        m, n = len(s), len(t)
        if m == 0: return True
        if m>n: return False
        l = 0
        for r in range(n):
            if t[r] == s[l]:
                l += 1
            if l == m: return True

        return False