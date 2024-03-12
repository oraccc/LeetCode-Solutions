class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        s = [0 for _ in range(n+1)]
        for i in range(1, n+1):
            s[i] = s[i-1] + nums[i-1]
        
        q = []
        min_len = n+1
        for i in range(n+1):
            curr = s[i]
            while q and curr - s[q[0]] >= k:
                min_len = min(min_len, i-q.pop(0))
            while q and curr <= s[q[-1]]:
                q.pop()
            q.append(i)

        if min_len == n+1: return -1
        else: return min_len