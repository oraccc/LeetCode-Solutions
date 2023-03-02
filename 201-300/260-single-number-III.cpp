vector<int> singleNumber(vector<int>& nums) {
    int total_xor = 0;
    for (const auto &num : nums) {
        total_xor ^= num;
    }
    int pos = 0;
    while (((total_xor >> pos) & 1) == 0) {
        ++pos;
    }
    int left_xor = 0, right_xor = 0;
    for (const auto &num : nums) {
        if (((num >> pos) & 1) == 0) {
            left_xor ^= num;
        }
        else {
            right_xor ^= num;
        }
    }

    return vector<int>{left_xor, right_xor};
}