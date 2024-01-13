class Solution {
    bool check(string &sub, string &s) {
        int k = s.size() / sub.size();
        string tmp;
        for (int i = 0; i < k; ++i) {
            tmp += sub;
        }
        return tmp == s;
    }
    
public:
    string gcdOfStrings(string str1, string str2) {
        string ans;
        if (str1.size() > str2.size()) {
            swap(str1, str2);
        }
        for (int i = str1.size(); i >= 1; --i) {
            string sub = str1.substr(0, i);
            if (check(sub, str1) && check(sub, str2)) {
                ans = sub;
                break;
            }
        }

        return ans;
    }
};