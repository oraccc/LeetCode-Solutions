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

// DFS with backtracking, use a help function to convert
vector<string> binaryTreePaths(TreeNode* root) {
    vector<string> ans;
    list<int> path;
    dfs(root, path, ans);
    return ans;
}

void dfs(TreeNode* &node, list<int> &path, vector<string> &ans) {
    if (node->left == nullptr && node->right == nullptr) {
        path.push_back(node->val);
        string s = convert(path);
        ans.push_back(s);
        return;
    }
    path.push_back(node->val);
    
    if (node->left != nullptr) {
        dfs(node->left, path, ans);
        path.pop_back();
    }
    if (node->right != nullptr) {
        dfs(node->right, path, ans);
        path.pop_back();
    }
    
}
string convert(const list<int> &path) {
    if (path.size() == 1) return to_string(path.front());
    else {
        string s = to_string(path.front());
        for (auto beg = path.begin(); beg != path.end(); ++beg){
            if (beg == path.begin()) continue;
            s = s + "->" + to_string(*beg);
        }
        return s;
    }
}