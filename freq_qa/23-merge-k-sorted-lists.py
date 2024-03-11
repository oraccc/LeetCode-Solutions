# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

ListNode.__lt__ = lambda a, b: a.val < b.val  # 让堆可以比较节点大小

class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        k = len(lists)
        if k == 0: return None
        h = [head for head in lists if head]
        heapify(h)
        dummy_head = ListNode()
        curr = dummy_head
        while h:
            node = heappop(h)
            if node.next:
                heappush(h, node.next)
            curr.next = node
            curr = curr.next
        return dummy_head.next
