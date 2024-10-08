vector<int> exchange(vector<int>& nums) {
    int left = 0, right = nums.size()-1;
    while (left < right) {
        while (left < right && nums[left] % 2 == 1) {
            ++left;
        }
        while (left < right && nums[right] % 2 == 0) {
            --right;
        }
        if (left < right) {
            int tmp = nums[left];
            nums[left] = nums[right];
            nums[right] = tmp;
        }
    }

    return nums;
}