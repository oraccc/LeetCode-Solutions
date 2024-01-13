string reverseLeftWords(string s, int n) {
    string left = s.substr(0, n);
    string right = s.substr(n);
    return right + left;
}