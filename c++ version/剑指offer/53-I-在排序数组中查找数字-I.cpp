int search(vector<int>& nums, int target) {
    if (nums.size() == 0) return 0;
    int low = lowerBound(nums, target);
    int high = upperBound(nums, target);

    if (low == nums.size() || nums[low] != target) return 0;
    return high - low;
}

int lowerBound(vector<int>& nums, int k) {
    int left = 0, right = nums.size();
    while (left < right) {
        int mid = left + (right - left)/2;
        if (nums[mid] >= k) {
            right = mid;
        }
        else left = mid + 1;
    }
    return left;
}

int upperBound(vector<int>& nums, int k) {
    int left = 0, right = nums.size();
    while (left < right) {
        int mid = left + (right - left)/2;
        if (nums[mid] > k) {
            right = mid;
        }
        else left = mid + 1;
    }
    return left;
}