vector<int> findDisappearedNumbers(vector<int>& nums) {
    vector<int> ans;
    int n = nums.size();
    vector<bool> flag(n+1, false);

    for (const auto &num : nums) {
        flag[num] = true;
    }
    for (int i = 1; i <= n; ++i) {
        if (flag[i] == false) {
            ans.push_back(i);
        }
    }
    return ans;
}