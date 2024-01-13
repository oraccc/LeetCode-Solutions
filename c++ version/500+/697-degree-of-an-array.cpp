int findShortestSubArray(vector<int>& nums) {
    int degree = 0;
    unordered_map<int, vector<int>> mp;

    for (int i = 0; i < nums.size(); ++i) {
        if (mp.count(nums[i]) == 0) {
            mp[nums[i]] = vector<int>{0, INT_MAX, 0};
        }
        ++mp[nums[i]][0];
        degree = max(degree, mp[nums[i]][0]);
        mp[nums[i]][1] = min(mp[nums[i]][1], i);
        mp[nums[i]][2] = i;
    }

    int ans = INT_MAX;
    for (const auto &pair : mp) {
        if (pair.second[0] == degree) {
            ans = min(ans, pair.second[2] - pair.second[1] + 1);
        }
    }
    return ans;
}