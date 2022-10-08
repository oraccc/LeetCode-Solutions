int maximumWealth(vector<vector<int>>& accounts) {
    int wealth = 0;
    int n = accounts.size();
    int m = accounts[0].size();
    for (int i = 0; i < n; ++i){
        int count = 0;
        for (int j = 0; j < m; ++j){
            count += accounts[i][j];
        }
        if (count > wealth) wealth = count;
    }

    return wealth;
    
}