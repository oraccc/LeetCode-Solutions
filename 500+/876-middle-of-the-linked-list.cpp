//Solution 1: distance between begin and end

ListNode* middleNode(ListNode* head) {
    double distance = 0.0;
    ListNode *end = head;
    while (end){
        end = end->next;
        distance += 1.0;
    }
        
    int half = ceil((distance -1.0)/2.0);

    for (int i = 0; i < half; ++i)
        head = head->next;

    return head;

}

//Solution 2: two pointers (fast & slow)

ListNode* middleNode(ListNode* head) {
    ListNode *slow, *fast;
    slow = fast = head;

    while(fast && fast->next){
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;
}