string reverseWords(string s) {
    stack<string> sc;
    int start = 0, end = 0;
    while (s.find_first_not_of(' ', end) != string::npos) {
        start = s.find_first_not_of(' ', end);
        end = s.find_first_of(' ', start);
        sc.push(s.substr(start, end-start));
    }

    string ans = "";
    while (!sc.empty()) {
        if (!ans.empty())
            ans += " ";
        ans += sc.top();
        sc.pop();
    }

    return ans;

}