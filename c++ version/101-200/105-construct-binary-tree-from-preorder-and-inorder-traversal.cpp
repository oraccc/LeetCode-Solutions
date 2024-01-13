class Solution {
    unordered_map<int, int> hash;
public:
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        for (int i = 0; i < inorder.size(); ++i) {
            hash[inorder[i]] = i;
        }
        return helper(preorder, 0, inorder.size()-1, 0);
    }

    TreeNode* helper(vector<int>& preorder, int inStart, int inEnd, int preStart) {
        if (inStart > inEnd) return nullptr;
        int rootVal = preorder[preStart];
        int inMid = hash[rootVal];
        int leftLen = inMid - inStart;
        TreeNode *root = new TreeNode(rootVal);
        root->left = helper(preorder, inStart, inMid - 1, preStart + 1);
        root->right = helper(preorder, inMid + 1, inEnd, preStart + 1 + leftLen);
        return root;
    }
};