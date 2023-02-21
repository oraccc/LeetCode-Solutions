class Solution {
    vector<TreeNode*> forest;
public:
    vector<TreeNode*> delNodes(TreeNode* root, vector<int>& to_delete) {
        unordered_set<int> s(to_delete.begin(), to_delete.end());
        root = helper(root, s);
        if (root) {
            forest.push_back(root);
        }

        return forest;
    }

    TreeNode* helper(TreeNode* root, unordered_set<int> &s) {
        if (root == nullptr) {
            return root;
        }
        root->left = helper(root->left, s);
        root->right = helper(root->right, s);
        if (s.count(root->val)) {
            if (root->left) {
                forest.push_back(root->left);
            }
            if (root->right) {
                forest.push_back(root->right);
            }
            root = nullptr;
        }

        return root;
    }


};