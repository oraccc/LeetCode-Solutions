vector<int> findAnagrams(string s, string p) {
    vector<int> ans;
    if (s.size() < p.size()) return ans;
    vector<int> freq(26, 0);
    for (const auto &c : p) {
        ++freq[c-'a'];
    }
    int left = 0, right = 0;
    while (right < s.size()) {
        --freq[s[right]-'a'];
        if (right-left+1 < p.size()) {
            ++right;
        }
        else {
            if (checkFreq(freq)) {
                ans.push_back(left);
            }
            ++freq[s[left]-'a'];
            ++left;
            ++right;
            
        }
    }
    return ans;
}

bool checkFreq(vector<int> &freq) {
    for (const auto &v : freq) {
        if (v != 0) return false;
    }
    return true;
}