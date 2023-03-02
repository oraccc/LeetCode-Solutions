vector<string> list;
vector<vector<string>> ans;
vector<vector<string>> partition(string s) {
    backtracking(0, s);
    return ans;
}

void backtracking(int start, string &s) {
    if (start == s.size()) {
        ans.push_back(list);
        return;
    }
    for (int i = start; i < s.size(); ++i) {
        if (isPalindrome(s, start, i)) {
            list.push_back(s.substr(start, i-start+1));
            backtracking(i+1, s);
            list.pop_back();
        }
    }
}

bool isPalindrome(string &s, int l, int r) {
    while (l < r) {
        if (s[l++] != s[r--]) 
            return false;
    }
    return true;
}