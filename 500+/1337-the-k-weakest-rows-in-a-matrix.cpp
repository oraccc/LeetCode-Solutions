vector<int> kWeakestRows(vector<vector<int>>& mat, int k) {
    vector<vector<int>> records;
    int n = mat.size();

    for (int i = 0; i < n; ++i)
    {
        int sum = accumulate(mat[i].begin(), mat[i].end(), 0);
        vector<int> row = {i, sum};
        records.push_back(row);
    }

    stable_sort(records.begin(), records.end(), 
        [](const auto &v1, const auto &v2) 
        {return v1[1] < v2[1];});

    vector<int> answer;

    for (int i = 0; i < k; ++i)
    {
        answer.push_back(records[i][0]);
    }

    return answer;
    
}