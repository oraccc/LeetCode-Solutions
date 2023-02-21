bool hasPathSum(TreeNode* root, int targetSum) {
    if (!root) return false;
    if (!root->left && !root->right) {
        if (targetSum == root->val) return true;
        else return false;
    }
    else return (hasPathSum(root->left, targetSum - root->val) || hasPathSum(root->right, targetSum - root->val));
}