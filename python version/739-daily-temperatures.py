class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        ans = [0] * n
        stack = []
        for i in range(n):
            while stack and temperatures[stack[-1]] < temperatures[i]:
                prev = stack.pop(-1)
                duration = i - prev
                ans[prev] = duration
            stack.append(i)

        return ans