//No refernece on string -> No backtracking
vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> ans;
    string s = "";
    dfs(root, s, ans);
    return ans;
}

void dfs(TreeNode* &node, string s, vector<string> &ans) {
    if (node->left == nullptr && node->right == nullptr) {
        s += to_string(node->val);
        ans.push_back(s);
        return;
    }
    s += to_string(node->val);
    if (node->left != nullptr) {
        dfs(node->left, s + "->", ans);
    }
    if (node->right != nullptr) {
        dfs(node->right, s + "->", ans);
    }
    
}