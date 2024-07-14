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
    
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans_dict = collections.defaultdict(list)
        for s in strs:
            counts = [0]*128
            for char in s:
                counts[ord(char)] += 1
            ans_dict[tuple(counts)].append(s)
        return list(ans_dict.values())