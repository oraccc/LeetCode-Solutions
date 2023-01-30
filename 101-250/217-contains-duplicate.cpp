bool containsDuplicate(vector<int>& nums) {
    set<int> s;
    for (const auto &num : nums) {
        auto ret = s.insert(num);
        if (ret.second == false) {
            return true;
        }
    }
    return false;
}