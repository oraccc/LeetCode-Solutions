class Solution {
    vector<vector<int>> ans;
    vector<int> path;
public:
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) return ans;
        backtracking(root, targetSum);
        return ans;
    }

    void backtracking(TreeNode* root, int target) {
        if (root == nullptr) return;
        path.push_back(root->val);
        if (!root->left && !root->right) {
            if (target == root->val) {
                ans.push_back(path);
            }
        }
        backtracking(root->left, target-root->val);
        backtracking(root->right, target-root->val);
        path.pop_back();
    }
};