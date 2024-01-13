// Recursive Solution
class Solution {
    vector<int> ans;
public:
    vector<int> preorderTraversal(TreeNode* root) {
        if (root == nullptr) return ans;
        helper(root);
        return ans;
    }

    void helper(TreeNode* root) {
        if (root == nullptr) return;
        ans.push_back(root->val);
        helper(root->left);
        helper(root->right);
    }
};

//Iteratively Version
class Solution {
    vector<int> ans;
public:
    vector<int> preorderTraversal(TreeNode* root) {
        if (root == nullptr) return ans;
        stack<TreeNode*> s;
        s.push(root);
        while (!s.empty()) {
            TreeNode* n = s.top();
            s.pop();
            ans.push_back(n->val);
            if (n->right) s.push(n->right);
            if (n->left) s.push(n->left);
        }
        return ans;
    }

};