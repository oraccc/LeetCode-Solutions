class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        answer = 0
        sign = 1
        curr = 0

        for c in s:
            if c.isnumeric():
                curr = curr*10+int(c)
            elif c == "+":
                answer += curr*sign
                curr = 0
                sign = 1
            elif c == "-":
                answer += curr*sign
                curr = 0
                sign = -1
            elif c == "(":
                stack.append(answer)
                stack.append(sign)
                answer = 0
                sign = 1
            elif c == ")":
                answer += curr*sign
                curr = 0
                
                answer *= stack.pop()
                answer += stack.pop()
        
        answer += curr*sign
        return answer