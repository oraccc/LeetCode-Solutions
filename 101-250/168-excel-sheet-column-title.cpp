string convertToTitle(int columnNumber) {
    string ans;
    while (columnNumber != 0) {
        columnNumber -= 1;
        char c = 'A' + (columnNumber % 26);
        ans = c + ans;
        columnNumber /= 26;
    }
    return ans;
}