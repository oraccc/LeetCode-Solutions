"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""

class Solution:
    def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
        if not head: return None
        curr = head
        # 1. 复制各节点，并构建拼接链表
        while curr:
            tmp = curr.next
            curr.next = Node(curr.val)
            curr.next.next = tmp
            curr = curr.next.next
        # 2. 构建各新节点的 random 指向
        curr = head
        while curr:
            if curr.random:
                curr.next.random = curr.random.next
            curr = curr.next.next
        # 3. 拆分两链表(记得复原原来的链表)
        curr = ans = head.next
        pre = head
        while curr.next:
            pre.next = pre.next.next
            curr.next = curr.next.next
            curr = curr.next
            pre = pre.next

        return ans
            
            
        