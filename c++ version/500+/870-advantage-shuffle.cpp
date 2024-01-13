vector<int> advantageCount(vector<int>& nums1, vector<int>& nums2) {
    multiset<int> s(nums1.begin(), nums1.end());
    vector<int> ans;
    for (const auto &num : nums2) {
        auto p = s.upper_bound(num);
        if (p == s.end()) {
            ans.push_back(*s.begin());
            s.erase(s.begin());
        }
        else {
            ans.push_back(*p);
            s.erase(p);
        }
    }
    return ans;
}