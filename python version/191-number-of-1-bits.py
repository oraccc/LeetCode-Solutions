class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            digit = n & 1
            count += digit|0
            n >>= 1
        return count
        