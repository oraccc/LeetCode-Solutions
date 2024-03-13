# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root: return None
        if root.val == p.val or root.val == q.val:
            return root
        left_ans = self.lowestCommonAncestor(root.left, p, q)
        right_ans = self.lowestCommonAncestor(root.right, p, q)
        if left_ans and right_ans:
            return root
        else:
            return left_ans if left_ans else right_ans
        