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
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        int low, high;
        if (q->val < p->val) {
            low = q->val; high = p->val;
        }
        else {
            low = p->val; high = q->val;
        }

        return helper(root, low, high);
    }

    TreeNode* helper(TreeNode* root, int low, int high) {
        if (root->val >= low && root->val <= high) return root;
        else if (root->val > high) return helper(root->left, low, high);
        else return helper(root->right, low, high);
    }
};