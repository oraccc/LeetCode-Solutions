int maxProduct(vector<string>& words) {
    int n = words.size(), maxSize = 0;
    vector<int> masks(n, 0);
    for (int i = 0; i < n; ++i) {
        for (const auto &c : words[i]) {
            masks[i] |= 1 << (c - 'a');
        }
        for (int j = 0; j < i; ++j) {
            if ((masks[i] & masks[j]) == 0) {
                maxSize = max(maxSize, int(words[i].size() * words[j].size()));
            }
        }
    }
    return maxSize;
}