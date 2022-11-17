unordered_map<int, vector<int>> memo;
vector<int> beautifulArray(int n) {
    auto it = memo.find(n);
    if (it != memo.end()) {
        return it -> second;
    }
    if (n == 1) {
        return {1};
    }
    
    int left = (n+1)/2, right = n/2;
    vector<int> odd = beautifulArray(left);
    vector<int> even = beautifulArray(right);
    vector<int> result;

    for (const int &i : odd) {
        result.push_back(2*i-1);
    }
    for (const int &i : even) {
        result.push_back(2*i);
    }
    memo.insert({n, result});
    return result;
}