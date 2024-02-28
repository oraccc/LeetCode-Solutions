class Solution:
    def intToRoman(self, num: int) -> str:
        chars = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
        nums = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]

        idx = 0
        ans = ""
        while num > 0:
            if num < nums[idx]:
                idx += 1
            else:
                num -= nums[idx]
                ans += chars[idx]

        return ans