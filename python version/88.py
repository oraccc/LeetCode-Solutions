class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        total_end = m+n-1
        first_end = m-1
        second_end = n-1
        while first_end >= 0 and second_end >= 0:
            if nums1[first_end] > nums2[second_end]:
                nums1[total_end] = nums1[first_end]
                first_end -= 1
                total_end -= 1
            else:
                nums1[total_end] = nums2[second_end]
                second_end -= 1
                total_end -= 1
        while second_end >= 0:
            nums1[total_end] = nums2[second_end]
            total_end -= 1
            second_end -= 1
