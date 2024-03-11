class Solution:
    def restoreIpAddresses(self, s: str) -> List[str]:
        ans = []
        curr = []

        def is_valid(s, start, end):
            if end > len(s): return False
            if end - start >= 2 and s[start] == "0": return False
            if int(s[start:end]) > 255: return False
            return True

        def backtracking(pos):
            if len(curr) == 4:
                if pos == len(s):
                    ans.append(".".join(curr))
                    return
                else:
                    return
            for k in range(1, 4):
                end = pos+k
                if is_valid(s, pos, pos+k):
                    curr.append(s[pos:pos+k])
                    backtracking(pos+k)
                    curr.pop()

        backtracking(0)
        return ans