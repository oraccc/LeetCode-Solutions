class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0.0: return 0.0
        if n < 0:
            x = 1/x
            n = -n
        ans = 1
        while n:
            if n & 1:
                ans = ans * x
            x = x*x
            n = n >> 1
        return ans