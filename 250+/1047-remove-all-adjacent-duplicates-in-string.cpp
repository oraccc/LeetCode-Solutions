string removeDuplicates(string s) {
    stack<char> sc;
    for (const auto &c : s) {
        if (!sc.empty() && sc.top() == c) {
            sc.pop();
        }
        else {
            sc.push(c);
        }
    }
    string ans = "";
    while (!sc.empty()) {
        ans += sc.top();
        sc.pop();
    }
    ans = string(ans.rbegin(), ans.rend());
    return ans;
}