int subarraySum(vector<int>& nums, int k) {
    int count = 0, psum = 0;
    unordered_map<int, int> hashmap;
    hashmap[0] = 1;
    for (const auto &num : nums) {
        psum += num;
        count += hashmap[psum-k];
        ++hashmap[psum];
    }
    return count;
}