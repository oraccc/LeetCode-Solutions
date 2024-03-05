class Solution:
    def summaryRanges(self, nums: List[int]) -> List[str]:
        if len(nums) == 0: return []
        i = 0
        j = 0
        ans = []
        while j+1 < len(nums):
            if nums[j+1] == nums[j]+1:
                j += 1
            else:
                if j == i:
                    ans.append(str(nums[i]))
                else:
                    ans.append(f"{nums[i]}->{nums[j]}")
                i = j+1
                j = i

        if j == i:
            ans.append(str(nums[i]))
        else:
            ans.append(f"{nums[i]}->{nums[j]}")

        return ans
