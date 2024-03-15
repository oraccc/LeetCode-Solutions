class Solution:
    def countDigitOne(self, n: int) -> int:
        ans = 0
        mul = 1
        while n >= mul:
            ans += n // (mul*10) * mul
            res = n % (mul*10)
            if res < mul:
                ans += 0
            elif res>=mul and res < 2*mul:
                ans += (res-mul+1)
            else:
                ans += mul
            mul *= 10
        return ans