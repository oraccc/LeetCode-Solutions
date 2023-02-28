bool isSubtree(TreeNode* root, TreeNode* subRoot) {
    if (root == nullptr) return false;
    return isSubtree(root->left, subRoot) || isSubtree(root->right, subRoot) || checkRoot(root, subRoot);
}

bool checkRoot(TreeNode* root, TreeNode* subRoot) {
    if (root == nullptr && subRoot == nullptr) return true;
    if (root == nullptr || subRoot == nullptr) return false;
    if (root->val != subRoot->val) return false;
    return checkRoot(root->left, subRoot->left) && checkRoot(root->right, subRoot->right);
}