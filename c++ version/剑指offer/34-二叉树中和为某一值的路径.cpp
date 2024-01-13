/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
    vector<vector<int>> ans;
public:
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        if (root == nullptr) return ans;
        vector<int> path;
        backtracking(root, target, path);

        return ans;
    }

    void backtracking(TreeNode* root, int target, vector<int> &path) {
        path.push_back(root->val);
        if (!root->left && !root->right) {
            if (root->val == target) {
                ans.push_back(path);
            }
        }

        if (root->left) {
            backtracking(root->left, target - root->val, path);
        }
        if (root->right) {
            backtracking(root->right, target - root->val, path);
        }
        path.pop_back();
    }
};