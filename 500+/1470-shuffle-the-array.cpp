vector<int> shuffle(vector<int>& nums, int n) {
    for (int i = 0; i < n; ++i) {
        nums[i] <<= 10;
        nums[i] |= nums[i+n];
    }

    for (int i = n-1; i >= 0; --i) {
        int x = nums[i] >> 10;
        int y = nums[i] & (1024-1);
        nums[2*i+1] = y;
        nums[2*i] = x;
    }

    return nums;
}