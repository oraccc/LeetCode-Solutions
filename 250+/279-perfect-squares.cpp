//dp

int numSquares(int n) {
    vector<int> dp(n+1, INT_MAX);
    dp[0] = 0;
    for (int i = 1; i <=n; ++i) {
        for (int j = 1; j * j <= i; ++j) {
            dp[i] = min(dp[i-j*j] + 1, dp[i]);
        }
    }
    return dp[n];
}

//BFS

int numSquares(int n) {
    queue<int> q;
    q.push(n);
    int level = 0;

    while (!q.empty()) {
        ++level;
        int qSize = q.size();
        while (qSize--) {
            int a = q.front();
            q.pop();
            for (int i = 1; i * i <= a; ++i) {
                int b = a - i*i;
                if (b == 0) {
                    return level;
                }
                else {
                    q.push(b);
                }
            }
        }
    }
    return level;
}