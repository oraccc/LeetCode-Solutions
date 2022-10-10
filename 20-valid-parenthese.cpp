bool isValid(string s) {
    stack<char> sc;
    for (const auto &c : s){
        switch (c) {
            case ')':
                if (sc.empty()) return false;
                if (sc.top() != '(') return false;
                sc.pop();
                break;
            case '}':
                if (sc.empty()) return false;
                if (sc.top() != '{') return false;
                sc.pop();
                break;
            case ']':
                if (sc.empty()) return false;
                if (sc.top() != '[') return false;
                sc.pop();
                break;
            default:
                sc.push(c);
                break;
        }
    }

    if (!sc.empty()) return false;
    return true;
}