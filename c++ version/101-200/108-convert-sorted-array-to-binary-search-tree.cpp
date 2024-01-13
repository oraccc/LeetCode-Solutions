TreeNode* sortedArrayToBST(vector<int>& nums) {
    return helper(0, nums.size(), nums);
}

TreeNode* helper(int start, int after, vector<int>& nums) {
    if (start == after) return nullptr;
    int mid = start + (after-start)/2;
    TreeNode* root = new TreeNode(nums[mid]);
    root->left = helper(start, mid, nums);
    root->right = helper(mid+1, after, nums);
    return root;
}