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

//divide in half and reverse

bool isPalindrome(ListNode* head) {
    if (!head->next) return true;
    ListNode *slow = head, *fast = head;
    while (fast->next && fast->next->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    slow->next = reverseList(slow->next);
    slow = slow->next;
    while (slow) {
        if (head->val != slow->val) {
            return false;
        }
        head = head->next;
        slow = slow->next;
    }
    return true;

}

ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr, *next;
    while (head) {
        next = head->next;
        head->next = prev;
        prev = head;
        head = next;
    }
    return prev;
}