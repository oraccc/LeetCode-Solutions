int findRepeatNumber(vector<int>& nums) {
    set<int> s;
    for (int i = 0; nums.size(); ++i) {
        auto ret = s.insert(nums[i]);
        if (!ret.second) return nums[i];
    }
    return -1;
}