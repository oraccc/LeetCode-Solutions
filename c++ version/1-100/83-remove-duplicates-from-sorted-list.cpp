ListNode* deleteDuplicates(ListNode* head) {
    if (head == nullptr) return head;
    ListNode *prev = head;
    ListNode *curr = prev->next;
    while (curr) {
        if (prev->val != curr->val) {
            prev = curr;
            curr = curr->next;
        }
        else {
            prev->next = curr->next;
            ListNode *tmp = curr;
            curr = prev->next;
            delete tmp;
        }
    }
    return head;
}