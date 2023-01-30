vector<int> nextGreaterElements(vector<int>& nums) {
    unordered_map<int, int> next;
    stack<int> s;
    int n = nums.size();
    for (int i = 0; i < n; ++i) {
        next[i] = -1;
    }

    for (int i = 0; i < 2*n; ++i) {
        int num = nums[i % n];
        while (!s.empty() && nums[s.top()] < num) {
            int pre = s.top();
            s.pop();
            next[pre] = num;
        }
        s.push(i % n);
    }
    vector<int> ans;
    for (int i = 0; i < n; ++i) {
        ans.push_back(next[i]);
    }
    
    return ans;
}