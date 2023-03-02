bool isAnagram(string s, string t) {
    vector<int> count(26, 0);
    if (s.size() != t.size()) {
        return false;
    }
    for (int i = 0; i < s.size(); ++i) {
        ++count[s[i] - 'a'];
        --count[t[i] - 'a'];
    }

    for (const auto &n : count) {
        if (n != 0) return false;
    }

    return true;
}