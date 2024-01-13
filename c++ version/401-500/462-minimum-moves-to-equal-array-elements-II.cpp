int minMoves2(vector<int>& nums) {
    int n = nums.size();
    sort(nums.begin(), nums.end());
    int mid = nums[n/2];
    int sum = 0;
    for (const auto &n : nums) {
        sum += abs(n-mid);
    }
    return sum;
}