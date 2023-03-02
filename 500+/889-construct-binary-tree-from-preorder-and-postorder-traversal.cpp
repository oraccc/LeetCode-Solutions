class Solution {
    unordered_map<int, int> hash;
public:
    TreeNode* constructFromPrePost(vector<int>& preorder, vector<int>& postorder) {
        for (int i = 0; i < postorder.size(); ++i) {
            hash[postorder[i]] = i;
        }

        return helper(preorder, postorder, 0, preorder.size()-1, 0, postorder.size()-1); 
    }

    TreeNode* helper(vector<int>& pre, vector<int> post, int preStart, int preEnd, int postStart, int postEnd) {
        if (preStart > preEnd) return nullptr;
        int rootVal = pre[preStart];
        TreeNode* root = new TreeNode(rootVal);
        if (preStart == preEnd) return root;
        int index = hash[pre[preStart+1]];
        int leftLen = index - postStart + 1;
        root->left = helper(pre, post, preStart+1, preStart+leftLen, postStart, index);
        root->right = helper(pre, post, preStart+leftLen+1, preEnd, index+1, postEnd);
        return root;
    }
};