int findJudge(int n, vector<vector<int>>& trust) {
    vector<bool> flag(n+1, false);
    vector<int> vote(n+1, 0);
    for (const auto &pair : trust) {
        if (flag[pair[0]] == false) {
            flag[pair[0]] = true;
        }
        ++vote[pair[1]];
    }
    for (int i = 1; i <= n; ++i) {
        if (flag[i] == false && vote[i] == n-1) {
            return i;
        }
    }
    return -1;
}