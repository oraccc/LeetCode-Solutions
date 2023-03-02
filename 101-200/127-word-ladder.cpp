// one-way BFS

int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> dict;
    for (const auto &word : wordList) {
        dict.insert(word);
    }
    if (dict.count(endWord) == 0) {
        return 0;
    }
    
    int ans = 1;
    dict.erase(beginWord);
    dict.erase(endWord);
    unordered_set<string> query{beginWord};
    bool found = false;
    while (!query.empty()) {
        ++ans;
        int q_size = query.size();
        unordered_set<string> tmp_q;
        for (const auto &w : query) {
            string s = w;
            for (int i = 0; i < s.size(); ++i) {
                char ch = s[i];
                for (int j = 0; j < 26; ++j) {
                    s[i] = j + 'a';
                    if (s == endWord) {
                        found = true;
                    }
                    if (dict.count(s)) {
                        tmp_q.insert(s);
                    }
                }
                s[i] = ch;
            }
        }
        if (found) break;
        for (const auto &w: tmp_q) {
            dict.erase(w);
        }
        query = tmp_q;
    }
    if (found) return ans;
    else return 0;
}

//two-way BFS

int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> dict;
    for (const auto &word : wordList) {
        dict.insert(word);
    }
    if (dict.count(endWord) == 0) {
        return 0;
    }
    int ans = 1;
    dict.erase(beginWord);
    dict.erase(endWord);
    unordered_set<string> query1{beginWord}, query2{endWord};
    bool found = false;
    while (!query1.empty()) {
        ++ans;
        unordered_set<string> q;
        for (const auto &w : query1) {
            string s = w;
            for (int i = 0; i < s.size(); ++i) {
                char ch = s[i];
                for (int j = 0; j < 26; ++j) {
                    s[i] = j + 'a';
                    if (query2.count(s)) {
                        found = true;
                    }
                    if (dict.count(s)) {
                        q.insert(s);
                    }
                }
                s[i] = ch;
            }
        }
        if (found) break;
        for (const auto &w: q) {
            dict.erase(w);
        }
        if (q.size() <= query2.size()) {
            query1 = q;
        }
        else {
            query1 = query2;
            query2 = q;
        }
    }

    if (found) return ans;
    else return 0;
}