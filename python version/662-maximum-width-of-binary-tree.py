# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def widthOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        max_width = 1
        if not root:
            return 0
        queue = [(root, 0)]

        while queue:
            n = len(queue)
            for _ in range(n):
                node, idx = queue.pop(0)
                if node.left:
                    queue.append((node.left, 2*idx))
                if node.right:
                    queue.append((node.right, 2*idx+1))
            
            if queue:
                max_width = max(max_width, queue[-1][1]-queue[0][1]+1)

        return max_width