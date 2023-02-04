int strStr(string haystack, string needle) {
    int k = -1, n = haystack.size(), m = needle.size();
    vector<int> next(m, -1);
    for (int j = 1, p = -1; j < m; ++j) {
        while (p > -1 && needle[p+1] != needle[j]) {
            p = next[p];
        }
        if (needle[p+1] == needle[j]) {
            ++p;
        }
        next[j] = p;
    }

    for (int i = 0; i < n; ++i) {
        while (k > -1 && needle[k+1] != haystack[i]) {
            k = next[k];
        }
        if (needle[k+1] == haystack[i]) {
            ++k;
        }
        if (k == m-1) {
            return i-m+1;
        }
    }
    return -1;
}