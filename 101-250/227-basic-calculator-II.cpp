int calculate(string s) {
    int i = 0;
    char op = '+';
    long long left = 0, right = 0;
    while (i < s.size()) {
        if (s[i] != ' ') {
            long long n = parseNum(s, i);
            switch (op) {
                case '+' : left += right; right = n; break;
                case '-' : left += right; right = -n; break;
                case '*' : right *= n; break;
                case '/' : right /= n; break;
            }
            if (i < s.size()) {
                op = s[i];
            }
        }
        ++i;
    }
    return left + right;
}

long long parseNum(string &s, int &i) {
    long long n = 0;
    while (i < s.size() && isdigit(s[i])) {
        n = n * 10 + (s[i] - '0');
        ++i;
    }
    return n;
}