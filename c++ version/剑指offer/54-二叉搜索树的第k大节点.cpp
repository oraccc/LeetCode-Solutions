/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
    int count = 0, ans = 0;
    bool found = false;
public:
    int kthLargest(TreeNode* root, int k) {
        helper(root, k);
        return ans;
    }

    void helper(TreeNode* root, int k) {
        if (root == nullptr || found == true) return;
        helper(root->right, k);
        if (++count == k) {
            ans = root->val;
            found = true;
            return;
        }
        helper(root->left, k);
    }
};