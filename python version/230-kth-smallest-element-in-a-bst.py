# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        left_num = self.count_node(root.left)
        if left_num == k-1:
            return root.val
        elif left_num > k-1:
            return self.kthSmallest(root.left, k)
        else:
            return self.kthSmallest(root.right, k-1-left_num)
    
    def count_node(self, root):
        if not root: return 0
        return 1+self.count_node(root.left)+self.count_node(root.right)
