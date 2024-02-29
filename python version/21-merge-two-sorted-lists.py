# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy_head = ListNode()
        curr = dummy_head
        l1 = list1
        l2 = list2
        while l1 and l2:
            if l1.val < l2.val:
                new_val = l1.val
                l1 = l1.next
            else:
                new_val = l2.val
                l2 = l2.next
            curr.next = ListNode(new_val)
            curr = curr.next
        if l1:
            curr.next = l1
        if l2:
            curr.next = l2 
        return dummy_head.next
            