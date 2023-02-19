bool isBalanced(TreeNode* root) {
    return helper(root) != -1;
}

int helper(TreeNode* root) {
    if (root == nullptr) return 0;
    int left = helper(root->left), right = helper(root->right);
    if (abs(left-right) > 1 || left == -1 || right == -1) {
        return -1;
    }
    return 1+max(left, right);
}