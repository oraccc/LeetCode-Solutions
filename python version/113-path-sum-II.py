# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> List[List[int]]:
        ans = []
        curr = []

        def dfs_helper(root, target):
            if not root:
                return
            curr.append(root.val)
            target -= root.val
            if not root.left and not root.right:
                if target == 0:
                    ans.append(curr[:])
            dfs_helper(root.left, target)
            dfs_helper(root.right, target)
            curr.pop()
        
        dfs_helper(root, targetSum)
        return ans
