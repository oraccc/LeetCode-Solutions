int findLHS(vector<int>& nums) {
    unordered_map<int, int> mp;
    for (const auto &num : nums) {
        if (mp.count(num) == 0) {
            mp[num] = 1;
        }
        else ++mp[num];
    }
    int ans = 0;
    for (const auto &num : nums) {
        if (mp.count(num+1) != 0) {
            ans = max(ans, mp[num] + mp[num+1]);
        }
    }
    return ans;
}