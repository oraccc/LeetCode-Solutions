class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        n = len(nums1)
        m = len(nums2)
        if n > m:
            nums1, nums2 = nums2, nums1
            n, m = m, n
        
        i_min = 0
        i_max = n
        while i_min <= i_max:
            i = (i_min + i_max)//2
            j = (m+n+1)//2 - i
            if i != 0 and j != m and nums1[i-1] > nums2[j]:
                i_max = i
            elif j != 0 and i != n and nums2[j-1] > nums1[i]:
                i_min = i+1
            else:
                if i == 0:
                    left_max = nums2[j-1]
                elif j == 0:
                    left_max = nums1[i-1]
                else:
                    left_max = max(nums1[i-1], nums2[j-1])

                if (m+n) % 2 == 1:
                    return left_max

                if i == n:
                    right_min = nums2[j]
                elif j == m:
                    right_min = nums1[i]
                else:
                    right_min = min(nums1[i], nums2[j])
                
                return (left_max + right_min)/2
        return 0