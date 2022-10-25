int singleNonDuplicate(vector<int>& nums) {
    int left = 0, right = nums.size() - 1;
    int mid;
    while (left < right) {
        mid = left + (right - left) / 2;
        if (mid % 2) {
            if (nums[mid] == nums[mid - 1]) left = mid + 1;
            else right = mid;
        }
        else {
            if (nums[mid] == nums[mid + 1]) left = mid + 1;
            else right = mid;
        }
    }

    return nums[left];
}