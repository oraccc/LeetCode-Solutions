# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:

        def helper(in_start, in_end, post_end):
            if in_start>in_end: return None
            val = postorder[post_end]
            node = TreeNode(val)
            in_pos = inorder.index(val)
            right_len = in_end-in_pos
            node.left = helper(in_start, in_pos-1, post_end-right_len-1)
            node.right = helper(in_pos+1, in_end, post_end-1)
            return node

        return helper(0, len(inorder)-1, len(postorder)-1)