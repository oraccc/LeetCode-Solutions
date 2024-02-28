# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        paths = []
        def backtracking(node, currSum):
            currSum = currSum*10 + node.val
            if not node.left and not node.right:
                paths.append(currSum)
                return
            if node.left:
                backtracking(node.left, currSum)
            if node.right:
                backtracking(node.right, currSum)
            return
        
        backtracking(root, 0)
        return sum(paths)
        