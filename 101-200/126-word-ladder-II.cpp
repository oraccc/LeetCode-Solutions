// timeout for complex cases

vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
    vector<vector<string>> ans;
    unordered_set<string> dict;
    for (const auto &word : wordList) {
        dict.insert(word);
    }
    if (dict.count(endWord) == 0) {
        return ans;
    }
    dict.erase(beginWord);
    dict.erase(endWord);
    unordered_set<string> query{beginWord};
    bool found = false;
    unordered_map<string, vector<string>> next;
    while (!query.empty()) {
        int q_size = query.size();
        unordered_set<string> tmp_q;
        for (const auto &w : query) {
            string s = w;
            for (int i = 0; i < s.size(); ++i) {
                char ch = s[i];
                for (int j = 0; j < 26; ++j) {
                    s[i] = j + 'a';
                    if (s == endWord) {
                        next[w].push_back(s);
                        found = true;
                    }
                    if (dict.count(s)) {
                        next[w].push_back(s);
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

    if (found) {
        vector<string> path = {beginWord};
        backtracking(beginWord, endWord, next, path, ans);
    }
    return ans;
}

void backtracking(const string &src, const string &dst, unordered_map<string, vector<string>> &next, vector<string> &path, vector<vector<string>> &ans) {
    if (src == dst) {
        ans.push_back(path);
        return;
    }
    for (const auto &s: next[src]) {
        path.push_back(s);
        backtracking(s, dst, next, path, ans);
        path.pop_back();
    }
}

// two-way search, still timeout for new test cases
vector<vector<string>> findLadders(string beginWord, string endWord, vector<string>& wordList) {
    vector<vector<string>> ans;
    unordered_set<string> dict;
    for (const auto &word : wordList) {
        dict.insert(word);
    }
    if (dict.count(endWord) == 0) {
        return ans;
    }
    dict.erase(beginWord);
    dict.erase(endWord);
    unordered_set<string> query1{beginWord}, query2{endWord};
    bool found = false, reversed = false;
    unordered_map<string, vector<string>> next;
    while (!query1.empty()) {
        unordered_set<string> q;
        for (const auto &w : query1) {
            string s = w;
            for (int i = 0; i < s.size(); ++i) {
                char ch = s[i];
                for (int j = 0; j < 26; ++j) {
                    s[i] = j + 'a';
                    if (query2.count(s)) {
                        reversed ? next[s].push_back(w) : next[w].push_back(s);
                        found = true;
                    }
                    if (dict.count(s)) {
                        reversed ? next[s].push_back(w) : next[w].push_back(s);
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
            reversed = !reversed;
            query1 = query2;
            query2 = q;
        }
    }

    if (found) {
        vector<string> path = {beginWord};
        backtracking(beginWord, endWord, next, path, ans);
    }
    return ans;
}

void backtracking(const string &src, const string &dst, unordered_map<string,vector<string>> &next, vector<string> &path, vector<vector<string>> &ans) {
    if (src == dst) {
        ans.push_back(path);
        return;
    }
    for (const auto &s: next[src]) {
        path.push_back(s);
        backtracking(s, dst, next, path, ans);
        path.pop_back();
    }
}