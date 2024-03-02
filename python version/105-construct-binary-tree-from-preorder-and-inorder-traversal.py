# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        
        def helper(pre_start, in_start, in_end):
            if in_start > in_end: return None
            val = preorder[pre_start]
            node = TreeNode(val)
            in_pos = inorder.index(val)
            len_left = in_pos - in_start
            node.left = helper(pre_start+1, in_start, in_pos-1)
            node.right = helper(pre_start+1+len_left, in_pos+1, in_end)
            return node
        
        return helper(0, 0, len(inorder)-1)