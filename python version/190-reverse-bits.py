class Solution:
    def reverseBits(self, n: int) -> int:
        ans = 0
        for i in range(32):
            digit = n & 1
            n >>= 1
            ans = ans | (digit << (31-i))
        
        return ans
        