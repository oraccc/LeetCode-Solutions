class Solution:
    def plusOne(self, digits: List[int]) -> List[int]:
        digits = digits[::-1]
        digits[0] += 1
        carry = 0
        for i in range(len(digits)):
            sum = digits[i] + carry
            carry, digits[i] = sum//10, sum%10
        if carry:
            digits.append(1)
        return digits[::-1]