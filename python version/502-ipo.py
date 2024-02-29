class Solution:
    def findMaximizedCapital(self, k: int, w: int, profits: List[int], capital: List[int]) -> int:
        max_heap = []
        pairs = list(zip(capital, profits))
        pairs.sort(key = lambda x: x[0])
        curr = 0

        for i in range(k):
            while curr < len(profits) and pairs[curr][0] <= w:
                heapq.heappush(max_heap, -pairs[curr][1])
                curr += 1

            if max_heap:
                w += -max_heap[0]
                heapq.heappop(max_heap)
            
            else: break

        return w