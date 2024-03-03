# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countNodes(self, root: Optional[TreeNode]) -> int:
        if root == None: return 0
        level = 0
        curr = root
        while curr.left:
            level += 1
            curr = curr.left

        def exist_node(count):
            bit = 1 << (level-1)
            curr = root
            while bit:
                if bit & count:
                    curr = curr.right
                else:
                    curr = curr.left
                bit >>= 1
            return curr != None
        
        left, right = 2**level, 2**(level+1)-1
        while left < right:
            mid = left + (right-left)//2 + 1
            if exist_node(mid):
                left = mid
            else:
                right = mid-1
        
        return left

        