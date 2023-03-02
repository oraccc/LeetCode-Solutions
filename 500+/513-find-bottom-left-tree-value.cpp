int findBottomLeftValue(TreeNode* root) {
    queue<TreeNode*> q;
    int ans;
    q.push(root);
    while (!q.empty()) {
        int len = q.size();
        for (int i = 0; i < len; ++i) {
            TreeNode* node = q.front();
            q.pop();
            if (i == 0) ans = node->val;
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
    }
    return ans;
}