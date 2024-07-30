class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        freq = collections.defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        uniques = list(freq.keys())
        heap = []
        for num in uniques:
            if len(heap) < k:
                heapq.heappush(heap, (freq[num], num))
            else:
                if freq[num] > heap[0][0]:
                    heapq.heappush(heap, (freq[num], num))
                    heapq.heappop(heap)
        
        ans = [x[1] for x in heap]
        return ans