# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def count_node(self, node):
        if not node: return 0
        return 1+self.count_node(node.left)+self.count_node(node.right)

    def findTargetNode(self, root: Optional[TreeNode], cnt: int) -> int:
        right_size = self.count_node(root.right)
        if cnt == right_size+1: return root.val
        elif cnt <= right_size:
            return self.findTargetNode(root.right, cnt)
        else:
            return self.findTargetNode(root.left, cnt-1-right_size)