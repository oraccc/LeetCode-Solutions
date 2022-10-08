bool isPalindrome(ListNode* head) {
    
    stack<int> s;
    ListNode *curr = head;
    while (curr) {
        s.push(curr->val);
        curr = curr->next;
    }

    curr = head;
    while (curr){
        if (curr->val == s.top()){
            s.pop();
            curr = curr->next;
        }

        else break;
    }

    return s.empty();
}