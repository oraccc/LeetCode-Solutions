int sumOfLeftLeaves(TreeNode* root) {
    if (!root->left && !root->right) return 0;
    return helper(root->left, root) + helper(root->right, root);
}

int helper(TreeNode* curr, TreeNode* father) {
    if (curr == nullptr) return 0;
    if (curr == father->left && (!curr->left && !curr->right))
        return curr->val;
    return helper(curr->left, curr) + helper(curr->right, curr);
}
