# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseLink(self, head):
        prev = None
        curr = head
        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp   
        return prev

    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        if left == 1: 
            prev = None
            reverse_start = head
        else:
            prev = head
            for i in range(left-2):
                prev = prev.next
            reverse_start = prev.next
        curr = reverse_start
        for i in range(right-left):
            curr = curr.next
        reverse_end = curr
        nxt = reverse_end.next
        reverse_end.next = None
        if prev:
            prev.next = self.reverseLink(reverse_start)
        else:
            head = self.reverseLink(reverse_start)
        
        curr = head
        while curr.next:
            curr = curr.next
        curr.next = nxt

        return head
        
            