vector<int> nextGreaterElements(vector<int>& nums) {
    stack<int> s;
    int n = nums.size();
    vector<int> ans(n, -1);

    for (int i = 0; i < 2*n; ++i) {
        int num = nums[i % n];
        while (!s.empty() && nums[s.top()] < num) {
            int pre = s.top();
            s.pop();
            ans[pre] = num;
        }
        s.push(i % n);
    }
    
    return ans;
}