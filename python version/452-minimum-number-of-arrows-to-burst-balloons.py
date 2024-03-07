class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key = lambda x: x[1])
        count = 1
        prev = 0
        for curr in range(1, len(points)):
            if points[curr][0] > points[prev][1]:
                count += 1
                prev = curr
            
        return count