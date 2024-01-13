// Recursive Solution
class Solution {
    vector<int> ans;
public:
    vector<int> postorderTraversal(TreeNode* root) {
        if (root == nullptr) return ans;
        helper(root);
        return ans;
    }

    void helper(TreeNode* root) {
        if (root == nullptr) return;
        helper(root->left);
        helper(root->right);
        ans.push_back(root->val);
    }
};

//Iteratively Version
class Solution {
    vector<int> ans;
public:
    vector<int> postorderTraversal(TreeNode* root) {
        if (root == nullptr) return ans;
        stack<TreeNode*> s;
        s.push(root);
        while (!s.empty()) {
            TreeNode* n = s.top();
            s.pop();
            ans.push_back(n->val);
            if (n->left) s.push(n->left);
            if (n->right) s.push(n->right);
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
};