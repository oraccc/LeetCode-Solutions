TreeNode* deleteNode(TreeNode* root, int key) {
    if (root == nullptr) return nullptr;
    TreeNode* temp;
    if (root->val > key) root->left = deleteNode(root->left, key);
    else if (root->val < key) root->right = deleteNode(root->right, key);
    else if (root->left && root->right) {
        temp = findMin(root->right);
        root->val = temp->val;
        root->right = deleteNode(root->right, root->val);
    }
    else {
        if (!root->left && !root->right) {
            delete root;
            return nullptr;
        }
        temp = root;
        if (root->left) root = root->left;
        else if (root->right) root = root->right;
        delete temp;
        
    }
    return root;
}