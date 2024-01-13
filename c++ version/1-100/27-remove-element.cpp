int removeElement(vector<int>& nums, int val) {
    int left = 0, right = 0;
    while (right != nums.size()) {
        if (nums[right] == val) {
            ++right;
        }
        else {
            nums[left] = nums[right];
            ++left;
            ++right;
        }
    }
    return left;
}
