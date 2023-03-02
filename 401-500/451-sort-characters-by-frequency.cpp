string frequencySort(string s) {
    unordered_map<int, int> mp;
    for (const auto &c : s)
        ++mp[c];
    
    vector<vector<int>> counts;
    for (const auto &item : mp)
        counts.push_back(vector<int>{item.second, item.first});
    
    sort(counts.begin(), counts.end(),
        [](const auto &v1, const auto &v2)
        {
            return (v1[0] > v2[0]) || (v1[0] == v2[0] && v1[1] < v2[1]);
        });
    
    string ret;
    for (const auto &v : counts) {
        int cnt = v[0];
        while (cnt > 0) {
            ret += char(v[1]);
            --cnt;
        }
    }
    return ret;
}