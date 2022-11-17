string makeGood(string s) {
    if (s.size() == 1) return s;
    stack<char> goodStack;
    for (int i = 0; i < s.size(); ++i) {
        if (goodStack.empty()) {
            goodStack.push(s[i]);
        }
        else {
            char topChar = goodStack.top();
            if (abs(topChar - s[i]) == 32) {
                goodStack.pop();
            }
            else {
                goodStack.push(s[i]);
            }
        }
        
    }

    string ans = "";
    while (!goodStack.empty()) {
        ans += goodStack.top();
        goodStack.pop();
    }
    ans = string(ans.rbegin(), ans.rend());
    return ans;
}