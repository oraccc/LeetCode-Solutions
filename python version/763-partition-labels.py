class Solution:
    def partitionLabels(self, s: str) -> List[int]:
        last = collections.defaultdict(int)
        for i in range(len(s)):
            last[s[i]] = i 
        
        prev_last = -1
        curr_last = last[s[0]]
        ans = []
        for i in range(len(s)):
            curr_last = max(curr_last, last[s[i]])
            if i == curr_last:
                ans.append(curr_last-prev_last)
                prev_last = curr_last
            
        return ans
