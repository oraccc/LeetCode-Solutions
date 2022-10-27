int removeDuplicates(vector<int>& nums) {
    int dupCount = 1;
    int left = 0, right = 1;
    while (right != nums.size()) {
        if (nums[left] == nums[right]) {
            if (dupCount < 2) {
                ++dupCount;
                nums[++left] = nums[right];
            }
            ++right;
        }
        else {
            nums[++left] = nums[right++];
            dupCount = 1;
        }
    }
    return left + 1;
}