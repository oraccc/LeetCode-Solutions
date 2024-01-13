ListNode* removeNthFromEnd(ListNode* head, int n) {
    ListNode *dummyHead = new ListNode(-1);
    dummyHead->next = head;
    ListNode *slow = dummyHead, *fast = dummyHead;
    for (int i = 0; i < n; ++i) {
        fast = fast->next;
    }
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next;
    }

    slow->next = slow->next->next;
    ListNode *newHead = dummyHead->next;
    delete dummyHead;
    return newHead;
}