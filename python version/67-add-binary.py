class Solution:
    def addBinary(self, a: str, b: str) -> str:
        a = a[::-1]
        b = b[::-1]
        m = len(a)
        n = len(b)
        carry = 0
        ans = ""
        for i in range(max(m, n)):
            a_bit = int(a[i]) if i < m else 0
            b_bit = int(b[i]) if i < n else 0
            bit_sum = a_bit + b_bit + carry
            carry = bit_sum // 2
            ans += str(bit_sum % 2)
        
        if carry:
            ans += "1"
        
        return ans[::-1]