# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def minDiffInBST(self, root: Optional[TreeNode]) -> int:
        self.diff = 100001
        self.prev = -1

        def helper(node):
            if not node: return
            helper(node.left)
            if self.prev != -1:
                self.diff = min(node.val-self.prev, self.diff)
            self.prev = node.val
            helper(node.right)

        helper(root)
        return self.diff