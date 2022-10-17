string minWindow(string s, string t) {
    vector<int> unseen_chars(128, 0);
    vector<bool> unseen_flags(128, false);

    for (int i = 0; i < t.size(); ++i) {
        unseen_flags[t[i]] = true;
        ++unseen_chars[t[i]];
    }

    int r = 0, l = 0, count = 0;
    int minStart = 0, minSize = INT_MAX;
    for (r = 0; r < s.size(); ++r) {
        if (unseen_flags[s[r]]) {
            --unseen_chars[s[r]];
            if (unseen_chars[s[r]] >= 0)
                ++count;        
        }

        while (count == t.size()) {
            if (r - l + 1 < minSize) {
                minStart = l;
                minSize = r - l + 1;
            }
            if (unseen_flags[s[l]]) {
                unseen_chars[s[l]] += 1;
                if (unseen_chars[s[l]] > 0)
                    --count;
            }
            ++l;
        }
    }

    if (minSize > s.size()) return "";
    else return s.substr(minStart, minSize);
}