bool isIsomorphic(string s, string t) {
    unordered_map<char, char> mp1;
    unordered_map<char, char> mp2;
    for (int i = 0; i < s.size(); ++i) {
        if (mp1.count(s[i]) == 0 && mp2.count(t[i]) == 0) {
            mp1[s[i]] = t[i];
            mp2[t[i]] = s[i];
        }
        else {
            if (mp1.count(s[i]) == 0 || mp2.count(t[i]) == 0 || mp1[s[i]] != t[i] || mp2[t[i]] != s[i]) {
                return false;
            }
        }
    }
    return true;
}

// first appear index
bool isIsomorphic(string s, string t) {
    vector<int> s_index(256, -1), t_index(256, -1);
    for (int i = 0; i < s.size(); ++i) {
        if (s_index[s[i]] != t_index[t[i]]) {
            return false;
        }
        s_index[s[i]] = t_index[t[i]] = i;
    }
    return true;
}