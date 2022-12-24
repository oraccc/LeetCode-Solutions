int majorityElement(vector<int>& nums) {
    int n = nums.size();
    int m = nums[0], count = 0;
    for (int i = 0; i < n; ++i) {
        if (nums[i] == m) {
            ++count;
        }
        else if (count == 0) {
            m = nums[i];
            ++count;
        }
        else {
            --count;
        }
    }

    return m;
}