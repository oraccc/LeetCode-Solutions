class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t): return False
        word_dict = {}
        for char in s:
            if char not in word_dict:
                word_dict[char] = 1
            else:
                word_dict[char] += 1
        
        for char in t:
            if char not in word_dict:
                return False
            word_dict[char] -= 1
            if word_dict[char] < 0:
                return False

        return True