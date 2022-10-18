#include <unordered_map>
#include <string>
#include <iostream>

using namespace std;

int lengthOfLongestSubstringKDistinct(string s, int k) {
    unordered_map<char, int> mc;
    int l = 0, r = 0;
    int len = 0;
    while (r < s.size()) {
        ++mc[s[r]];
        while (mc.size() > k) {
            if (mc.find(s[l]) != mc.end())
                --mc[s[l]];
            if (mc[s[l]] == 0) mc.erase(s[l]);
            ++l;
        }
        len = max(len, r - l + 1);
        ++r;
    }

    return len;
}

int main() {
    string s = "kcebea";
    int k = 4;
    cout << lengthOfLongestSubstringKDistinct(s, k);
}