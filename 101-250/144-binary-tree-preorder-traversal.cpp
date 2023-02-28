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