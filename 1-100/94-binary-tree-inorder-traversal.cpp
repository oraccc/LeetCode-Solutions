// Recursive Solution
class Solution {
    vector<int> ans;
public:
    vector<int> inorderTraversal(TreeNode* root) {
        if (root == nullptr) return ans;
        helper(root);
        return ans;
    }

    void helper(TreeNode* root) {
        if (root == nullptr) return;
        helper(root->left);
        ans.push_back(root->val);
        helper(root->right);
    }
};

//Iteratively Version
