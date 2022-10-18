bool search(vector<int>& nums, int target) {
    int l = 0, r = nums.size() - 1;
    int mid;
    while (l <= r) {
        mid = l + (r - l) / 2;
        if (nums[mid] == target) return true;
        if (nums[mid] == nums[l]) ++l;
        else if (nums[mid] == nums[r]) --r;
        else if (nums[mid] < nums[r]) {
            if (target > nums[mid] && target <= nums[r]) 
                l = mid + 1;
            else 
                r = mid - 1;
        }
        else {
            if (nums[mid] > target && target >= nums[l])
                r = mid - 1;
            else 
                l = mid + 1;
        }
    }
    return false;
}