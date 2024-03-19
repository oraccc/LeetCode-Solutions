# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findMode(self, root: Optional[TreeNode]) -> List[int]:
        self.ans = set()
        self.max_count = 1
        self.curr_count = 1
        self.prev = None

        def helper(node):
            if not node:
                return
            helper(node.left)

            if self.prev:
                if self.prev.val == node.val:
                    self.curr_count += 1
                else:
                    self.curr_count = 1
                if self.curr_count > self.max_count:
                    self.max_count = self.curr_count
                    self.ans = set()
                    self.ans.add(node.val)
                elif self.curr_count == self.max_count:
                    self.ans.add(node.val)
            else:
                self.ans.add(node.val)
            self.prev = node
            helper(node.right)

        helper(root)

        return list(self.ans)
