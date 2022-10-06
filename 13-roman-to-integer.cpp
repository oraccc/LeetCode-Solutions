#include <string>
#include <map>
using namespace std;

class Solution {
public:
    int romanToInt(string s) {
        map<char, int> mp = {
            {'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100},
            {'D', 500},{'M', 1000}
        };
        int sum = 0;
        for (int i = 0; i < s.size() - 1; ++i){
            if (mp[s[i+1]] > mp[s[i]])
                sum -= mp[s[i]];
            else sum += mp[s[i]];
        }

        sum += mp[s[s.size()-1]];

        return sum;
    }
};