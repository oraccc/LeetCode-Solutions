class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        n = len(s)
        length = 0
        for i in range(n-1, -1, -1):
            if s[i] != " ":
                length += 1
            elif length != 0:
                break
        return length