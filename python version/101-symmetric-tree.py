# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def helper(self, left, right):
        if left == None and right == None: return True
        elif left == None or right == None: return False
        elif left.val != right.val: return False
        else:
            return self.helper(left.right, right.left) and self.helper(left.left, right.right)
        
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        return self.helper(root.left, root.right)