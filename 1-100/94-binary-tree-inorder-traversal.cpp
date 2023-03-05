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
class Solution {
    vector<int> ans;
public:
    vector<int> inorderTraversal(TreeNode* root) {
        if (root == nullptr) return ans;
        stack<TreeNode*> s;
        TreeNode* curr = root;
        s.push(root);
        curr = curr->left;
        while (curr || !s.empty()) {
            while (curr) {
                s.push(curr);
                curr = curr->left;
            }
            curr = s.top();
            s.pop();
            ans.push_back(curr->val);
            curr = curr->right;
        }

        return ans;
    }

};