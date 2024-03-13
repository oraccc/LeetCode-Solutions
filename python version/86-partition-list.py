# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def partition(self, head: Optional[ListNode], x: int) -> Optional[ListNode]:
        small_dummy = ListNode(0)
        big_dummy = ListNode(0)
        small = small_dummy
        big = big_dummy
        curr = head
        while curr:
            if curr.val < x:
                small.next = curr
                small = small.next
            else:
                big.next = curr
                big = big.next
            curr = curr.next

        small.next = big_dummy.next
        # 否则会有环
        big.next = None
        return small_dummy.next
                