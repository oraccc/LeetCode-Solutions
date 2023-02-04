int longestPalindrome(string s) {
    map<char, int> mp;
    for (const auto &c : s) {
        if (mp.count(c) == 0) {
            mp[c] = 1;
        }
        else ++mp[c];
    }

    int count = 0;
    bool flag = false;
    for (const auto &pair : mp) {
        count += pair.second/2*2;
        if (flag == false && pair.second%2 == 1) {
            count += 1;
            flag = true;
        }
    }
    return count;
}