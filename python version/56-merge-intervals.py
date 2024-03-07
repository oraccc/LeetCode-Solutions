class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key = lambda x:x[0])
        ans = []
        curr_start = intervals[0][0]
        curr_end = intervals[0][1]
        for i in range(1, len(intervals)):
            if curr_end >= intervals[i][0]:
                curr_end = max(intervals[i][1], curr_end)
            else:
                ans.append([curr_start, curr_end])
                curr_start = intervals[i][0]
                curr_end = intervals[i][1]
        ans.append([curr_start, curr_end])

        return ans