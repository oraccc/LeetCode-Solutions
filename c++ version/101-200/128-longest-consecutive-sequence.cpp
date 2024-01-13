int longestConsecutive(vector<int>& nums) {
    unordered_set<int> hash;
    for (const auto &num : nums) {
        hash.insert(num);
    }
    int ans = 0;
    while (!hash.empty()) {
        int curr = *(hash.begin());
        hash.erase(curr);
        int next = curr+1, prev = curr-1;
        while (hash.count(next)) {
            hash.erase(next++);
        }
        while (hash.count(prev)) {
            hash.erase(prev--);
        }
        ans = max(ans, next - prev - 1);
    }
    return ans;
}