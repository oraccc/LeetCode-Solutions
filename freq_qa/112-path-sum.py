# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def hasPathSum(self, root: Optional[TreeNode], targetSum: int) -> bool:
        if root == None: return False
        target = targetSum - root.val
        if not root.left and not root.right:
            if target == 0: return True
            else: return False
        else:
            return self.hasPathSum(root.left, target) or self.hasPathSum(root.right, target)