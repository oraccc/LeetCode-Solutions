class Solution:
    def nextGreaterElement(self, nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        hash_dict = {}
        for num in nums2:
            while stack and stack[-1] < num:
                prev = stack.pop(-1)
                hash_dict[prev] = num
            stack.append(num)
        
        ans = []
        for num in nums1:
            ans.append(hash_dict.get(num, -1))
        return ans
