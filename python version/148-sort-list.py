# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head == None or head.next == None: return head
        slow = fast = head
        prev = None
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next
        tmp = prev.next
        prev.next = None
        sorted_left = self.sortList(head)
        sorted_right = self.sortList(tmp)
        dummy_head = ListNode(0)
        curr = dummy_head
        l = sorted_left
        r = sorted_right
        while l and r:
            if l.val < r.val:
                curr.next = l
                l = l.next
            else:
                curr.next = r
                r = r.next
            curr = curr.next
        if l:
            curr.next = l
        elif r:
            curr.next = r

        return dummy_head.next