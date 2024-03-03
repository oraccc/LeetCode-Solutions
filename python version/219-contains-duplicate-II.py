class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        record = {}
        for i in range(len(nums)):
            if nums[i] not in record:
                record[nums[i]] = i
            elif i - record[nums[i]] <= k: return True
            else:
                record[nums[i]] = i
        
        return False