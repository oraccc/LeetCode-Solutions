int sumNumbers(TreeNode* root) {
    int sum = 0;
    backtracking(root, 0, sum);
    return sum;
}

void backtracking(TreeNode* root, int curr, int &sum) {
    curr = curr*10 + root->val;
    if (!root->left && !root->right) {
        sum += curr;
        return;
    }
    if (root->left) {
        backtracking(root->left, curr, sum);
    }
    if (root->right) {
        backtracking(root->right, curr, sum);
    }
}