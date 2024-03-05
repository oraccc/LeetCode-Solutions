"""
# Definition for a Node.
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
"""

from typing import Optional
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        visited = {}
        
        def bfs(node):
            if node == None: return
            clone = Node(node.val, [])
            visited[node] = clone
            queue = [node]

            while queue:
                tmp = queue.pop(0)
                for each in tmp.neighbors:
                    if each not in visited:
                        visited[each] = Node(each.val, [])
                        queue.append(each)
                    visited[tmp].neighbors.append(visited[each])

            return clone
        
        return bfs(node)