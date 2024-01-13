class Solution {
    unordered_map<int, int> hash;
public:
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        for (int i = 0; i < inorder.size(); ++i) {
            hash[inorder[i]] = i;
        }
        return helper(inorder, postorder, 0, inorder.size()-1, 0, postorder.size()-1);
    }

    TreeNode* helper(vector<int>& inorder, vector<int>& postorder, int inStart, int inEnd, int postStart, int postEnd) {
        if (postStart > postEnd) return nullptr;
        int nodeVal = postorder[postEnd];
        TreeNode* root = new TreeNode(nodeVal);
        int index = hash[nodeVal];
        int leftLen = index - inStart;
        root->left = helper(inorder, postorder, inStart, index-1, postStart, postStart+leftLen-1);
        root->right = helper(inorder, postorder, index+1, inEnd, postStart+leftLen, postEnd-1);
        return root;
    }
};