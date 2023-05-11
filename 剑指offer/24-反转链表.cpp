ListNode* reverseList(ListNode* head) {
    ListNode* prev = nullptr;
    ListNode* tmp;
    while (head) {
        tmp = head->next;
        head->next = prev;
        prev = head;
        head = tmp;
    }

    return prev;
}