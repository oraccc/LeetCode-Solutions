# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def zigzagLevelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        ans = []
        if not root: return ans
        queue = [root]
        flag = False
        while queue:
            size = len(queue)
            tmp = []
            for _ in range(size):
                prev = queue.pop(0)
                tmp.append(prev.val)
                if prev.left:
                    queue.append(prev.left)
                if prev.right:
                    queue.append(prev.right)
            if flag:
                tmp = tmp[::-1]
            flag = not flag
            ans.append(tmp)
        return ans