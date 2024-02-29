# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        self.max_sum = -1001
        def dfs(node):
            if node == None:
                return 0
            left_value = dfs(node.left)
            right_value = dfs(node.right)
            # 记录全局最大值
            self.max_sum = max(self.max_sum, node.val+left_value+right_value)
            # 返回可以连接上的最大值
            return max(0, node.val+max(left_value, right_value))
        dfs(root)
        return self.max_sum