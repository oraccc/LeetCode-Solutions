TreeNode* sortedListToBST(ListNode* head) {
    return helper(head, nullptr);
}

TreeNode* helper(ListNode* head, ListNode* after) {
    if (head == after) return nullptr;
    ListNode *slow = head, *fast = head;
    while (fast->next != after && fast->next->next != after) {
        slow = slow->next;
        fast = fast->next->next;
    }
    TreeNode* root = new TreeNode(slow->val);
    root->left = helper(head, slow);
    root->right = helper(slow->next, after);

    return root;
}