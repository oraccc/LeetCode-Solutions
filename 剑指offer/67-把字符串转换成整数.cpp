int strToInt(string str) {
    if (str.empty()) return 0;
    int start = 0;
    long long num = 0;
    while (str[start] == ' ')
        ++start;
    str = str.substr(start);
    if (str.empty()) return 0;
    bool neg = false;
    if (str[0] == '+') {
        str = str.substr(1);
        if (str.empty()) return 0;
    }
    else if (str[0] == '-') {
        str = str.substr(1);
        neg = true;
        if (str.empty()) return 0;
    }
    if (!isdigit(str[0])) return 0;
    else {
        for (int i = 0; i < str.size(); ++i) {
            if (!isdigit(str[i])) {
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
            num += (str[i] - '0');
        }
    }

    if (neg) num = -num;
    if (num > INT_MAX) return INT_MAX;
    else if (num < INT_MIN) return INT_MIN;
    return num;
    
}