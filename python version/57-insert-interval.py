class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        ans = []
        new_start = newInterval[0]
        new_end = newInterval[1]
        inserted = False
        for interval in intervals:
            if new_start > interval[1]:
                ans.append(interval)
            elif new_end < interval[0]:
                if not inserted:
                    inserted = True
                    ans.append([new_start, new_end])
                ans.append(interval)
                
            else:
                new_start = min(new_start, interval[0])
                new_end = max(new_end, interval[1])
        if not inserted:
            ans.append([new_start, new_end])
        return ans
        