# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sortedArrayToBST(self, nums: List[int]) -> Optional[TreeNode]:
        n = len(nums)
        if n == 0: return None
        root = TreeNode(nums[n//2])
        root.left = self.sortedArrayToBST(nums[0:n//2])
        root.right = self.sortedArrayToBST(nums[n//2+1:n])
        return root