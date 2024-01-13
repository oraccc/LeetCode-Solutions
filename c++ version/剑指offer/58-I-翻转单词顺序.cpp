string reverseWords(string s) {
    int left = 0, right = s.size()-1;
    while (left < right && s[left] == ' ') {
        ++left;
    }
    while (left < right && s[right] == ' ') {
        --right;
    }
    stack<string> stk;
    string ans, word;
    while (left <= right) {
        if (s[left] != ' ') {
            word += s[left];
        }
        else if (s[left] == ' ' && !word.empty()) {
            stk.push(word);
            word = "";
        }
        ++left;
    }
    stk.push(word);

    while (!stk.empty()) {
        ans += stk.top();
        stk.pop();
        if (!stk.empty()) {
            ans += ' ';
        }
    }
    return ans;
}