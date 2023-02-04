bool checkInclusion(string s1, string s2) {
    if (s1.size() > s2.size()) return false;
    vector<int> freq(26, 0);
    for (const auto &c : s1) {
        ++freq[c-'a'];
    }
    int left = 0, right = 0;
    while (right < s2.size()) {
        --freq[s2[right]-'a'];
        if (right-left+1 < s1.size()) {
            ++right;
        }
        else {
            if (checkFreq(freq)) return true;
            else {
                ++freq[s2[left]-'a'];
                ++left;
                ++right;
                
            }
        }
    }
    return false;
}

bool checkFreq(vector<int> &freq) {
    for (const auto &v : freq) {
        if (v != 0) return false;
    }
    return true;
}