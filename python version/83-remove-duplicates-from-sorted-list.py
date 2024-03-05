# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head: return None
        prev = head
        nxt = head
        while nxt:
            if nxt.val == prev.val:
                nxt = nxt.next
            else:
                prev.next = nxt
                prev = prev.next
                nxt = nxt.next
        prev.next = nxt
        
        return head