int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    int mid;
    while (left < right) {
        mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < nums[right]) {
            if (target > nums[mid] && target <= nums[right]) left = mid + 1;
            else right = mid;
        }
        else if (nums[mid] > nums[right]) {
            if (target < nums[mid] && target >= nums[left]) right = mid;
            else left = mid + 1;
        }
    }
    if (nums[left] == target) return left;
    else return -1;
}