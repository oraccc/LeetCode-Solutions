ListNode* swapPairs(ListNode* head) {
    ListNode* dummy = new ListNode(-1, head);
    ListNode* prev = dummy, *curr = head;
    while (curr && curr->next) {
        prev->next = curr->next;
        curr->next = curr->next->next;
        prev->next->next = curr;
        prev = curr;
        curr = curr->next;
    }
    ListNode* newHead = dummy->next;
    delete dummy;
    return newHead;
}