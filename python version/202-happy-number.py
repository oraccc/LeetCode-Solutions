class Solution:
    def isHappy(self, n: int) -> bool:
        if n == 1: return True
        s = set()
        while n != 1 and n not in s:
            s.add(n)
            tmp = 0
            while n != 0:
                tmp += (n%10)**2
                n = n // 10
            n = tmp
        if n == 1:return True
        else: return False