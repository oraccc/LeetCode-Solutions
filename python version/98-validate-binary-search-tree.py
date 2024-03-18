# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def find_most_left(self, root):
        while root.left:
            root = root.left
        return root

    def find_most_right(self, root):
        while root.right:
            root = root.right
        return root

    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
        left = root.left
        if left:
            most_right = self.find_most_right(left)
        right = root.right
        if right:
            most_left = self.find_most_left(right)

        if left:
            if left.val >= root.val or most_right.val >= root.val:
                return False
        if right:
            if right.val <= root.val or most_left.val <= root.val:
                return False

        return self.isValidBST(root.left) and self.isValidBST(root.right)