# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next: return head
        dummy_head = ListNode()
        dummy_head.next = head
        count = 0
        prev, curr = dummy_head, head
        while curr:
            count += 1
            if count == 2:
                tmp = curr.next
                curr.next = prev.next
                prev.next = curr
                curr.next.next = tmp
                prev = curr.next
                curr = tmp
                count = 0
            else:
                curr = curr.next
        
        return dummy_head.next