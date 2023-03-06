class Solution {
    TreeNode* start = nullptr;
public:
    bool findTarget(TreeNode* root, int k) {
        if (start == nullptr) start = root;
        if (root == nullptr) return false;
        if (search(root, k - root->val)) return true;
        return findTarget(root->left, k) || findTarget(root->right, k);
    }

    bool search(TreeNode* node, int k) {
        TreeNode* root = start;
        while(root) {
            if (root->val == k) return root == node ? false : true;
            else if (root->val > k) root = root->left;
            else if (root->val < k) root = root->right;
        }

        return false;
    }
};