class Solution:
    def isPalindrome(self, x: int) -> bool:
        if x < 0: return False
        if x!=0 and x%10 == 0: return False
        original = x
        new = 0
        while (original > new):
            new = 10 * new+original%10
            original = original // 10

        return original == new or original == new//10