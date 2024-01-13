int getMinimumDifference(TreeNode* root) {
    TreeNode* prev = nullptr;
    int diff = INT_MAX;
    inorder(root, prev, diff);
    return diff;
}

void inorder(TreeNode* root, TreeNode* &prev, int &diff) {
    if (root == nullptr) return;
    inorder(root->left, prev, diff);
    if (prev) {
        diff = min(root->val - prev->val, diff);
    }
    prev = root;
    inorder(root->right, prev, diff);
}