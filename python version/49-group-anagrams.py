class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = {}
        for s in strs:
            sort_s = "".join(sorted(s))
            if sort_s in ans:
                ans[sort_s].append(s)
            else:
                ans[sort_s] = [s]
        return list(ans.values())