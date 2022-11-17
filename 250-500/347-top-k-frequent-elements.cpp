// c++ sort function
vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> mp;
    for (const int &num : nums)
        ++mp[num];
    
    vector<vector<int>> counts;
    for (const auto &item : mp) {
        counts.push_back(vector<int>{item.second, item.first});
    }

    sort(counts.begin(), counts.end(), 
        [](const auto &v1, const auto &v2)
        {
            return v1[0] > v2[0];
        });
    
    vector<int> ans;
    for (int i = 0; i < k; ++i)
    {
        ans.push_back(counts[i][1]);
    }

    return ans;
}

// bucket sort

vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> mp;
    int max_count = 0;
    for (const int &num : nums)
        max_count = max(max_count, ++mp[num]);

    vector<vector<int>> buckets(max_count + 1);
    for (const auto &p : mp) {
        buckets[p.second].push_back(p.first);
    }

    vector<int> ans;
    for (int i = max_count; i >= 0 && ans.size() < k; --i) {
        for (const int & num : buckets[i]) {
            ans.push_back(num);
            if (ans.size() == k) break;
        }
    }

    return ans;
}