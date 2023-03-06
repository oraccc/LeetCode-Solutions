class Solution {
    TreeNode* ans = nullptr;
    TreeNode* prev = nullptr;
public:
    TreeNode* increasingBST(TreeNode* root) {
        helper(root);
        return ans;
    }

    void helper(TreeNode* root) {
        if (root == nullptr) return;
        helper(root->left);
        if (ans == nullptr) {
            ans = root;
        }
        else {
            prev->right = root;
        }
        prev = root;
        root->left = nullptr;
        helper(root->right);
    }
};