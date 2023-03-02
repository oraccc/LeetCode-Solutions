ListNode* sortList(ListNode* head) {
    if (head == nullptr || head->next == nullptr) return head;
    ListNode *slow = head, *fast = head;
    ListNode *temp = nullptr;
    while (fast && fast->next) {
        temp = slow;
        slow = slow->next;
        fast = fast->next->next;
    }
    temp->next = nullptr;
    ListNode *left = sortList(head);
    ListNode *right = sortList(slow);

    ListNode *dummyHead = new ListNode(-1);
    dummyHead->next = head;
    ListNode *curr = dummyHead;
    while (left && right) {
        if (left->val <= right->val) {
            curr->next = left;
            left = left->next;
        }
        else {
            curr->next = right;
            right = right->next;
        }
        curr = curr->next;
    }

    if (left) curr->next = left;
    if (right) curr->next = right;
    return dummyHead->next;
}