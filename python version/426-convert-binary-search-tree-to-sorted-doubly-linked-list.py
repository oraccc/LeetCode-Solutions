"""
# Definition for a Node.
class Node:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
"""
class Solution:
    def treeToDoublyList(self, root: 'Node') -> 'Node':
        if not root:
            return None
        self.prev = None

        def helper(node):
            if not node:
                return
            helper(node.left)
            if self.prev:
                self.prev.right = node
                node.left = self.prev
            else:
                self.head = node
            self.prev = node
            helper(node.right)
        
        helper(root)

        self.head.left = self.prev
        self.prev.right = self.head

        return self.head

            
        