struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
    int n1 = 0, n2 = 0, carry = 0;
    int sum = 0, result = 0;

    ListNode *dummyHead = new ListNode(0);
    ListNode *curr = dummyHead;

    while (l1 || l2 || carry) {
        n1 = (l1 ? l1->val : 0);
        n2 = (l2 ? l2->val : 0);
        sum = n1 + n2 + carry;
        carry = ((sum / 10) ? 1 : 0);
        result = sum % 10;
        curr->next = new ListNode(result);
        curr = curr -> next;
        l1 = (l1 ? l1->next : nullptr);
        l2 = (l2 ? l2->next : nullptr);
    }

    return dummyHead->next;
}