int diameterOfBinaryTree(TreeNode* root) {
    int diameter = 0;
    int tmp = helper(root, diameter);
    return diameter;
}

int helper(TreeNode* root, int &diameter) {
    if (root == nullptr) return 0;
    int left = helper(root->left, diameter), right = helper(root->right, diameter);
    diameter = max(left+right, diameter);
    return 1+max(left,right);
}