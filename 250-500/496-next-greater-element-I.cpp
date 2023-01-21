vector<int> nextGreaterElement(vector<int>& nums1, vector<int>& nums2) {
    unordered_map<int, int> next;
    stack<int> s;
    
    for (const auto &num : nums2) {
        while (!s.empty() && s.top() < num) {
            int pre = s.top();
            s.pop();
            next[pre] = num;
        }
        s.push(num);
    }
    vector<int> ans;
    for (const auto &num : nums1) {
        ans.push_back((next.count(num) == 0) ? -1 : next[num]);
    }
    return ans;
}