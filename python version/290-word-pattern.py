class Solution:
    def wordPattern(self, pattern: str, s: str) -> bool:
        mp1 = {}
        mp2 = {}
        words = s.split(" ")
        if len(pattern) != len(words): return False
        for i in range(len(pattern)):
            if pattern[i] not in mp1 and words[i] not in mp2:
                mp1[pattern[i]] = words[i]
                mp2[words[i]] = pattern[i]
            elif pattern[i] not in mp1 or words[i] not in mp2 or mp1[pattern[i]] != words[i] or mp2[words[i]] != pattern[i]:
                return False
        
        return True