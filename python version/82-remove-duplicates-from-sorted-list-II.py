# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head or not head.next: return head
        dummyHead = ListNode()
        dummyHead.next = head
        flag = False
        prev = dummyHead
        curr = head
        while curr:
            if not curr.next:
                if not flag:
                    prev.next = curr
                    curr = curr.next
                else:
                    prev.next = None
                    curr = curr.next
            else:
                if curr.val == curr.next.val:
                    flag = True
                    curr = curr.next
                else:
                    if flag:
                        prev.next = curr.next
                        curr = prev.next
                    else:
                        prev = curr
                        curr = curr.next
                    flag = False
        
        return dummyHead.next