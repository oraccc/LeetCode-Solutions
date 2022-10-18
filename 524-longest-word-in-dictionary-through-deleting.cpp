string findLongestWord(string s, vector<string>& dictionary) {
    sort(dictionary.begin(), dictionary.end(), 
        [](const string &s1, const string &s2) {
            return s1.size() > s2.size() || (s1.size() == s2.size() && s1 < s2);
        });
    for (int n = 0; n < dictionary.size(); ++n) {
        int i = 0;
        for (int j = 0; j < s.size(); ++j) {
            if (s[j] == dictionary[n][i]) ++i;
            if (i == dictionary[n].size()) return dictionary[n];
        }
    }

    return "";
}