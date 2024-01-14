class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        mp1 = {}
        mp2 = {}
        for i in range(len(s)):
            if s[i] not in mp1 and t[i] not in mp2:
                mp1[s[i]] = t[i]
                mp2[t[i]] = s[i]
            elif s[i] not in mp1 or t[i] not in mp2 or mp1[s[i]] != t[i] or mp2[t[i]] != s[i]:
                return False

        return True