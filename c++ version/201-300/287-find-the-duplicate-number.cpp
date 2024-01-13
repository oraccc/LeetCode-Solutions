int findDuplicate(vector<int>& nums) {
    int left = 1, right = nums.size();
    int mid = 0;
    int count = 0;
    while (left < right) {
        mid = left + (right-left)/2;
        count = 0;
        for (const auto &num : nums) {
            if (num <= mid) ++count;
        }
        if (count > mid) {
            right = mid;
        }
        else {
            left = mid + 1;
        }
    }
    return left;
}