ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr;
    ListNode *curr = head;
    ListNode *tmp = nullptr;

    while(curr){
        tmp = curr->next;
        curr->next = prev;
        prev = curr;
        curr = tmp;
    }

    return prev;
}

// Recursion
ListNode* reverseList(ListNode* head, ListNode* prev = nullptr) {
    if (!head) return prev;
    ListNode* next = head->next;
    head->next = prev;
    return reverseList(next, head);     
}