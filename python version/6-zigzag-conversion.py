class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows == 1: return s
        ans = ["" for i in range(numRows)]
        idx, flag = 0, 1
        for char in s:
            ans[idx] += char
            idx += flag
            if idx == 0 or idx == numRows-1:
                flag = -flag
        
        return "".join(ans)

