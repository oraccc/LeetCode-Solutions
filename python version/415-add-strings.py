class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        ans = ""
        num1 = num1[::-1]
        num2 = num2[::-1]
        len1 = len(num1)
        len2 = len(num2)
        carry = 0
        i = 0
        while i < len1 or i < len2:
            digit1 = int(num1[i]) if i < len1 else 0
            digit2 = int(num2[i]) if i < len2 else 0
            ans += str((digit1 + digit2 + carry) % 10)
            carry = (digit1 + digit2 + carry) // 10
            i += 1
        if carry:
            ans += "1"

        return ans[::-1]