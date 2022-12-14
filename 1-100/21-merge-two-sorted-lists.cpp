struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};


ListNode* mergeTwoLists(ListNode* list1, ListNode* list2) {
    ListNode *dummyHead = new ListNode(0);
    ListNode *curr = dummyHead;

    while (list1 && list2)
    {
        if (list1->val < list2->val){
            curr->next = new ListNode(list1->val);
            list1 = list1->next;
        }
        else{
            curr->next = new ListNode(list2->val);
            list2 = list2->next;
        }
        curr = curr->next;
    }
    if (list1){
        curr->next = list1;
    }
    else if (list2){
        curr->next = list2;
    }

    return dummyHead->next;
}