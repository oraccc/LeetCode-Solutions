class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        count = [0 for _ in range(128)]
        left = 0
        length = 0
        for right in range(len(s)):
            count[ord(s[right])] += 1
            while count[ord(s[right])] > 1:
                count[ord(s[left])] -= 1
                left += 1
            length = max(length, right-left+1)

        return length