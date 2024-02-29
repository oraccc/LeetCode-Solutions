class Solution:
    def mySqrt(self, x: int) -> int:
        if x == 0: return 0
        l, r = 1, x
        while l < r:
            # 思考最后只有两个数字的情况，需要强制向上取整
            mid = l + (r-l)//2 + 1
            if mid * mid == x: return mid
            elif mid * mid < x: l = mid
            else: r = mid - 1

        return l