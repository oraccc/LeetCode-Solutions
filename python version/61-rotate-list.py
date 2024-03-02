# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def rotateRight(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if head == None or head.next == None: return head
        list_len = 0
        curr = head
        while curr:
            curr = curr.next
            list_len += 1
        k = k % list_len
        if k == 0: return head
        curr = head
        for _ in range(list_len-k-1):
            curr = curr.next
        new_head = curr.next
        curr.next = None
        curr = new_head
        while curr.next:
            curr = curr.next
        curr.next = head
        return new_head