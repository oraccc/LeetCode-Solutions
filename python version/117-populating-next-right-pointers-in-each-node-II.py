"""
# Definition for a Node.
class Node:
    def __init__(self, val: int = 0, left: 'Node' = None, right: 'Node' = None, next: 'Node' = None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next
"""

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root: return root
        stack = [root]

        while stack:
            n = len(stack)
            prev = None
            curr = None
            for i in range(n):
                if not prev:
                    prev = stack.pop(0)
                else:
                    curr = stack.pop(0)
                    prev.next = curr
                    prev = curr
                if prev.left:
                    stack.append(prev.left)
                if prev.right:
                    stack.append(prev.right)
        return root

        