int myAtoi(string s) {
    if (s.empty()) return 0;
    int start = 0;
    long long num = 0;
    while (s[start] == ' ')
        ++start;
    s = s.substr(start);
    if (s.empty()) return 0;
    bool neg = false;
    if (s[0] == '+') {
        s = s.substr(1);
        if (s.empty()) return 0;
    }
    else if (s[0] == '-') {
        s = s.substr(1);
        neg = true;
        if (s.empty()) return 0;
    }
    if (!isdigit(s[0])) return 0;
    else {
        for (int i = 0; i < s.size(); ++i) {
            if (!isdigit(s[i])) {
                if (neg) num = -num;
                if (num > INT_MAX) return INT_MAX;
                else if (num < INT_MIN) return INT_MIN;
                return num;
            }
            if (num > INT_MAX) {
                if (neg) return INT_MIN;
                else return INT_MAX;
            }
            num *= 10;
            num += (s[i] - '0');
        }
    }

    if (neg) num = -num;
    if (num > INT_MAX) return INT_MAX;
    else if (num < INT_MIN) return INT_MIN;
    return num;
}