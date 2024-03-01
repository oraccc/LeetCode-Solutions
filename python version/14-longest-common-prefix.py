class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        ans = ""
        min_len = min(len(s) for s in strs)
        idx = 0
        while idx < min_len:
            char = ""
            for i in range(len(strs)):
                if not char:
                    char = strs[i][idx]
                elif strs[i][idx] != char:
                    return ans
            idx += 1
            ans += char
        
        return ans