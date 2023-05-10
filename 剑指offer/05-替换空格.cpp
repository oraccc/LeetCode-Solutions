string replaceSpace(string s) {
    int i = 0;
    while (i < s.size()) {
        if (s[i] != ' ') ++i;
        else {
            s.insert(i, "%20");
            s.erase(i+3, 1);
            i = i+3;
        }
    }
    return s;
}