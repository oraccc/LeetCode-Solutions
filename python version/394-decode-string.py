class Solution:
    def decodeString(self, s: str) -> str:
        stack = []
        i = 0
        ans = ""
        while i < len(s):
            if s[i] >= "0" and s[i] <= "9":
                times = 0
                while s[i] >= "0" and s[i] <= "9":
                    times = times*10 + int(s[i])
                    i += 1
                stack.append(times)
            elif s[i] != "]":
                stack.append(s[i])
                i += 1
            else:
                curr = ""
                while stack[-1] != "[":
                    curr = stack.pop() + curr
                stack.pop()
                times = stack.pop()
                stack.append(curr*times)
                i += 1
        ans = "".join(stack)
        return ans

