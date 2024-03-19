# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.diameter = 0

        def helper(node):
            if not node:
                return 0
            left_count = helper(node.left)
            right_count = helper(node.right)
            self.diameter = max(self.diameter, left_count+right_count)
            return 1 + max(left_count, right_count)    
        helper(root)
        return self.diameter