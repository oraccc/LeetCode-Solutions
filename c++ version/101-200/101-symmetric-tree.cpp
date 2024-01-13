bool isSymmetric(TreeNode* root) {
    return check(root->left, root->right);
}

bool check(TreeNode* left, TreeNode* right) {
    if (!left && !right) return true;
    else if (!left || !right) return false;
    else if (left->val != right->val) return false;
    else return (check(left->left, right->right) && check(left->right, right->left));
}