# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def bstFromPreorder(self, preorder: List[int]) -> Optional[TreeNode]:

        if not preorder:
            return
        n = len(preorder)
        val = preorder[0]
        root = TreeNode(val)
        start = 1
        while start < n:
            if preorder[start] < val:
                start += 1
            else:
                break
        root.left = self.bstFromPreorder(preorder[1:start])
        root.right = self.bstFromPreorder(preorder[start:])
        return root