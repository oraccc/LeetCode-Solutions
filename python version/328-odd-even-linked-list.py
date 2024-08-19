# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        odd_head = odd = head
        even_head = even = head.next

        while even and even.next:
            odd.next = even.next
            even.next = even.next.next
            odd.next.next = even

            odd = odd.next
            even = even.next
        
        odd.next = even_head
        return odd_head

