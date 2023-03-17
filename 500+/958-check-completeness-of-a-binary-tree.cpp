bool isCompleteTree(TreeNode* root) {
    bool flag = false;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int size = q.size();
        for (int i = 0; i < size; ++i) {
            TreeNode* node = q.front();
            q.pop();
            if (!node->left) flag = true;
            else {
                if (flag) return false;
                q.push(node->left);
            }
            if (!node->right) flag = true;
            else {
                if (flag) return false;
                q.push(node->right);
            }
        }
    }
    return true;
}