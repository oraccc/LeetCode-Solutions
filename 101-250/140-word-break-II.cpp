vector<string> wordBreak(string s, vector<string>& wordDict) {
    unordered_set<string> wordSet;
    int maxLength = 0;
    for (const auto &word : wordDict) {
        wordSet.insert(word);
        maxLength = max(maxLength, int(word.size()));
    }
    cout << maxLength << endl;
    vector<string> ans;
    string curr = "";
    backtracking(s, wordSet, maxLength, ans, 0, curr);

    return ans;
}

void backtracking(string &s, unordered_set<string> &wordSet, int &maxLength, 
    vector<string> &ans, int startPos, string curr) {
    if (startPos == s.size()) {
        ans.push_back(curr.substr(0, curr.size() - 1));
        return;
    }

    for (int endPos = startPos + 1; endPos <= s.size(); ++endPos) {
        if ((endPos - startPos) > maxLength) break;
        if (wordSet.find(s.substr(startPos, endPos - startPos)) != wordSet.end()) {
            backtracking(s, wordSet, maxLength, ans, endPos, curr + s.substr(startPos, endPos - startPos) + " ");
        }
    }
}
