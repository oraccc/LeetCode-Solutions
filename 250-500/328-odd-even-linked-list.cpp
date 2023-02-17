ListNode* oddEvenList(ListNode* head) {
    if (head == nullptr || head->next == nullptr) return head;
    ListNode *oddHead = head, *odd = head;
    ListNode *evenHead = head->next, *even = head->next;
    while (even && even->next) {
        odd->next = even->next;
        even->next = odd->next->next;
        odd = odd->next;
        even = even->next;
    }
    odd->next = evenHead;
    return head;
}