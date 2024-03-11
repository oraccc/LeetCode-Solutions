# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        if not head.next: return True

        slow, fast = head, head
        while fast.next and fast.next.next:
            slow = slow.next
            fast = fast.next.next
        halfway = slow
        
        halfway.next = self.reverseList(halfway.next)
        first_start, second_start = head, halfway.next

        while second_start:
            if first_start.val != second_start.val:
                return False
            else:
                first_start = first_start.next
                second_start = second_start.next

        return True


    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            tmp = curr.next
            curr.next = prev
            prev = curr
            curr = tmp
        return prev