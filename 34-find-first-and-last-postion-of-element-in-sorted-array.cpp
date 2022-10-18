vector<int> searchRange(vector<int>& nums, int target) {
    if (nums.size() == 0) 
        return vector<int>{-1, -1};
    int low = lower_bound(nums, target);
    int high = upper_bound(nums, target) - 1;
    if (low == nums.size() || nums[low] != target) 
        return vector<int>{-1, -1};

    return vector<int>{low, high};
}

int lower_bound(vector<int>& nums, int target) {
    int l = 0, r = nums.size();
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] >= target)
            r = mid;
        else l = mid + 1;
    }
    return r;
}

int upper_bound(vector<int>& nums, int target) {
    int l = 0, r = nums.size();
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (nums[mid] > target)
            r = mid;
        else l = mid + 1;
    }
    return r;
}