class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        heap = nums[:k]
        heapq.heapify(heap)

        for i in range(k, len(nums)):
            heapq.heappush(heap, nums[i])
            heapq.heappop(heap)

        return heap[0]
    

# quick_sort
    
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quick_sort_k(nums, k):
            pivot = random.choice(nums)
            big, equal, small = [], [], []
            for num in nums:
                if num > pivot:
                    big.append(num)
                elif num < pivot:
                    small.append(num)
                else:
                    equal.append(num)

            if k <= len(big):
                return quick_sort_k(big, k)
            if k > len(big) + len(equal):
                return quick_sort_k(small, k-len(big)-len(equal))

            return pivot
        return quick_sort_k(nums, k)