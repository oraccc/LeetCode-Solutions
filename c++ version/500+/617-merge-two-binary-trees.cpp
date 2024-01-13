TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
    if (root1 && root2) {
        TreeNode* node = new TreeNode();
        node->val = root1->val + root2->val;
        node->left = mergeTrees(root1->left, root2->left);
        node->right = mergeTrees(root1->right, root2->right);
        return node;
    }
    else {
        return root1 ? root1 : root2;
    }
    
}