bool isNumber(string s) {
    if (s.empty()) return false;
    int start = 0, end = 0;
    while (s[start] == ' ') {
        ++start;
    }
    for (end = start; end < s.size(); ++end) {
        if (s[end] == ' ') break;
    }
    for (int i = end; i < s.size(); ++i) {
        if (s[i] != ' ') return false;
    }
    s = s.substr(start, end-start);
    if (s.find('e') != string::npos) {
        int pos = s.find('e');
        string preE = s.substr(0, pos);
        string afterE = s.substr(pos + 1);
        if ((isDit(preE) || isInt(preE)) && isInt(afterE)) return true;
        else return false;
    }
    else if (s.find('E') != string::npos) {
        int pos = s.find('E');
        string preE = s.substr(0, pos);
        string afterE = s.substr(pos + 1);
        if ((isDit(preE) || isInt(preE)) && isInt(afterE)) return true;
        else return false;
    }
    else {
        return (isDit(s) || isInt(s));
    }
}

bool isInt(string s) {
    if (s.size() == 0) return false;
    if (s[0] == '+' || s[0] == '-') {
        s = s.substr(1);
        if (s.size() == 0) return false;
    }

    for (int i = 0; i < s.size(); ++i) {
        if (!isdigit(s[i])) return false;
    }
    return true;
}

bool isDit(string s) {
    if (s.size() == 0) return false;
    if (s[0] == '+' || s[0] == '-') {
        s = s.substr(1);
        if (s.size() == 0) return false;
    }
    int pos = s.find('.');
    if (pos == string::npos) return false;
    string beforeDot = s.substr(0, pos);
    string afterDot = s.substr(pos + 1);
    // cout << beforeDot << " " << afterDot << endl;
    // cout << isAllNum(beforeDot) << " " <<isAllNum(afterDot);
    if ((isAllNum(beforeDot) && afterDot.empty()) || (isAllNum(beforeDot) && isAllNum(afterDot)) || (beforeDot.empty() && isAllNum(afterDot)))
        return true;
    else return false;
}

bool isAllNum(string s) {
    if (s.size() == 0) return false;
    for (int i = 0; i < s.size(); ++i) {
        if (!isdigit(s[i])) return false;
    }
    return true;
}