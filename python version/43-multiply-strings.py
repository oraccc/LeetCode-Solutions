class Solution:
    def multiply(self, num1: str, num2: str) -> str:

        def cal_single(nums, s):
            result = 0
            carry = 0
            times = 0
            s = int(s)
            if s == 0:
                return 0
            for i in range(len(nums)-1, -1, -1):
                digit = int(nums[i])
                mul = (digit*s+carry) % 10
                carry = (digit*s+carry) // 10
                result += (10**times) * mul
                times += 1
            if carry:
                result += (10**times) * carry

            return result

        ans = 0
        for i in range(len(num2)):
            ans = ans*10+cal_single(num1, num2[i])

        return str(ans)