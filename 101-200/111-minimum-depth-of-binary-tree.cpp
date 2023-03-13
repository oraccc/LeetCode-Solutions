int minDepth(TreeNode* root) {
    if (root == nullptr) return 0;
    int level = 0;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        ++level;
        int n = q.size();
        for (int i = 0; i < n; ++i) {
            TreeNode* node = q.front();
            q.pop();
            if (!node->left && !node->right) {
                return level;
            }
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return level;
}