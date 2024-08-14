# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def convertBST(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None
        
        self.prev = None

        def helper(node):
            if not node:
                return

            helper(node.right)

            if self.prev:
                node.val += self.prev.val
            self.prev = node

            helper(node.left)

        helper(root)

        return root

                