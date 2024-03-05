class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        letter_count = {}
        for i in range(len(magazine)):
            letter_count[magazine[i]] = letter_count.get(magazine[i], 0) + 1

        for i in range(len(ransomNote)):
            letter_count[ransomNote[i]] = letter_count.get(ransomNote[i], 0) - 1
            if letter_count[ransomNote[i]] < 0: return False

        return True