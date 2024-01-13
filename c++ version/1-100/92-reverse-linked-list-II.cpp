ListNode* reverseBetween(ListNode* head, int left, int right) {
    ListNode* dummy = new ListNode(-1);
    dummy->next = head;
    ListNode* pre = dummy;
    for (int i = 0; i < left-1; ++i) {
        pre = pre->next;
    }
    ListNode* start = pre->next;
    for (int j = 0; j < right-left; ++j) {
        ListNode* tmp = pre->next;
        pre->next = start->next;
        start->next = start->next->next;
        pre->next->next = tmp;
    }

    ListNode* newHead = dummy->next;
    delete dummy;
    return newHead;
}