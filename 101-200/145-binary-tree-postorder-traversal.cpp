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