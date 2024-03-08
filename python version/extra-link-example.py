from typing import *

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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

    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        if not head.next or k == 1: return head
        dummy_head = ListNode()
        dummy_head.next = head
        prev, curr = dummy_head, head
        count = 0
        while curr:
            count += 1
            if count == k:
                tmp = curr.next
                curr.next = None
                prev.next = self.reverseLink(prev.next)
                while prev.next:
                    prev = prev.next
                prev.next = tmp
                count = 0
                curr = tmp
            else:
                curr = curr.next
        
        return dummy_head.next

if __name__ == "__main__":
    head = ListNode(1)
    head.next = ListNode(2)
    head.next.next = ListNode(3)
    head.next.next.next = ListNode(4)
    head.next.next.next.next = ListNode(5)

    s = Solution()
    new_head = s.reverseKGroup(head, 3)
    curr = new_head 
    while curr:
        print(curr.val, end=" ")
        curr = curr.next