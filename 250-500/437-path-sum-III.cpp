int pathSum(TreeNode* root, int targetSum) {
    if (root == nullptr) return 0;
    else return rootSum(root, targetSum) + pathSum(root->left, targetSum) + pathSum(root->right, targetSum);
}

int rootSum(TreeNode* root, long long target) {
    if (root == nullptr) return 0;
    int count = 0;
    if (root->val == target) ++count;
    count += rootSum(root->left, target - root->val);
    count += rootSum(root->right, target - root->val);
    return count;
}