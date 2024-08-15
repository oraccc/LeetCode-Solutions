# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:

        def helper(node):
            if not node:
                return 0
            left_height = helper(node.left)
            right_height = helper(node.right)

            if abs(left_height-right_height) > 1 or left_height == -1 or right_height == -1:
                return -1
            else:
                return 1+max(left_height, right_height)
        
        return helper(root) != -1