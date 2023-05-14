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
    unordered_map<int, int> inHash;
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        for (int i = 0; i < inorder.size(); ++i) {
            inHash[inorder[i]] = i;
        }

        return helper(preorder, 0, 0, inorder.size()-1);
    }

    TreeNode* helper(vector<int>& preorder, int preStart, int inStart, int inEnd) {
        if (inStart > inEnd) return nullptr;
        int val = preorder[preStart];
        int inMid = inHash[val];
        int leftLen = inMid - inStart;
        TreeNode* root = new TreeNode(val);
        root->left = helper(preorder, preStart+1, inStart, inMid-1);
        root->right = helper(preorder, preStart+1+leftLen, inMid+1, inEnd);
        return root;
    }
};