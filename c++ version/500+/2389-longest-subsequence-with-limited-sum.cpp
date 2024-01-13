vector<int> answerQueries(vector<int>& nums, vector<int>& queries) {
    int n = queries.size();
    sort(nums.begin(), nums.end());
    vector<int> ans(n, 0);
    for (int i = 0; i < n; ++i) {
        int sum = 0;
        int count = 0;
        for (int j = 0; j < nums.size(); ++j) {
            sum += nums[j];
            if (sum > queries[i]) {
                break;
            }
            else ++count;
        }
        ans[i] = count;
    }
    return ans;
}