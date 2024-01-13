char firstUniqChar(string s) {
    unordered_map<char, int> mp;
    for (int i = 0; i < s.size(); ++i) {
        if (mp.count(s[i]) == 0) mp[s[i]] = 1;
        else mp[s[i]] += 1;
    }

    for (int i = 0; i < s.size(); ++i) {
        if (mp[s[i]] == 1) return s[i];
    }

    return ' ';
}