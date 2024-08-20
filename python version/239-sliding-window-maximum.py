class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        deque = []
        n = len(nums)
        ans = []
        for i in range(n):
            if i < k:
                while deque and deque[-1] < nums[i]:
                    deque.pop()
                deque.append(nums[i])
                if i == k-1:
                    ans.append(deque[0])
            else:
                if deque and deque[0] == nums[i-k]:
                    deque.pop(0)
                while deque and deque[-1] < nums[i]:
                    deque.pop()
                deque.append(nums[i])
                ans.append(deque[0])
        return ans