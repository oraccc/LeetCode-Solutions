// Solution 1
int lengthOfLongestSubstring(string s) {
    if (s.empty()) return 0;
    int max_length = 1, count = 1;
    int prev = 0, curr = 1;
    set<char> sc = {s[0]};
    while (curr < s.size())
    {
        auto ret = sc.insert(s[curr]);
        if (ret.second)
        {
            curr += 1;
            count += 1;
            if (count > max_length) max_length = count;
        }
        else
        {
            count = 1;
            prev += 1;
            curr = prev + 1;
            sc = {s[prev]};
        }
    }

    return max_length;
    
}

// Solution 2: Two Pointers (left and right)
int lengthOfLongestSubstring(string s) {
    if (s.empty()) return 0;

    int max_length = 1;
    int left = 0, right = 1;
    unordered_set<char> sc;
    sc.insert(s[0]);
    while (left < s.size() && right < s.size()){
        auto ret = sc.insert(s[right]);
        if (ret.second){
            max_length = max(max_length, right - left + 1);
            right += 1;
        }
        else{
            while (sc.count(s[right])){
                sc.erase(s[left]);
                ++left;
            }
        }
    }

    return max_length;
    
}