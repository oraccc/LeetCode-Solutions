class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        pre_sum = 0
        count = 0
        prefix_dict = collections.defaultdict(int)
        prefix_dict[0] = 1
        for num in nums:
            pre_sum += num
            count += prefix_dict[pre_sum-k]
            prefix_dict[pre_sum] += 1
        return count