class Solution:
    def evalRPN(self, tokens: List[str]) -> int:
        stack = []
        for each in tokens:
            if each not in ["+", "-", "*", "/"]:
                stack.append(int(each))
            elif each == "+":
                second = stack.pop()
                first = stack.pop()
                stack.append(first + second)
            elif each == "-":
                second = stack.pop()
                first = stack.pop()
                stack.append(first - second)
            elif each == "*":
                second = stack.pop()
                first = stack.pop()
                stack.append(first * second)
            elif each == "/":
                second = stack.pop()
                first = stack.pop()
                # 注意不要使用//， 会向下取整
                stack.append(int(first / second))

        return stack[-1]