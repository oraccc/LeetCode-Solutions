struct Comp {
    bool operator() (ListNode* l1, ListNode* l2) {
        return l1->val > l2->val;
    }
};

ListNode* mergeKLists(vector<ListNode*>& lists) {
    if (lists.empty()) return nullptr;
    priority_queue<ListNode*, vector<ListNode*>, Comp> q;
    for (const auto &list : lists) {
        if (list != nullptr) {
            q.push(list);
        }
    }
    ListNode* dummyHead = new ListNode(-1), *curr = dummyHead;
    while (!q.empty()) {
        curr->next = q.top();
        q.pop();
        curr = curr->next;
        if (curr->next != nullptr) {
            q.push(curr->next);
        }
    }

    return dummyHead -> next;
}