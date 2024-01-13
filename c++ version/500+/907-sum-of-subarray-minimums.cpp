int sumSubarrayMins(vector<int>& arr) {
    int n = arr.size();
    vector<int> dp(n, 0);
    stack<int> closestNum;
    int sum = 0;
    int mod = 1000000007;
    for (int i = 0; i < n; ++i) {
        while (!closestNum.empty() && arr[closestNum.top()] > arr[i]) {
            closestNum.pop();
        }
        if (!closestNum.empty()) {
            int j = closestNum.top();
            dp[i] = dp[j] + (i-j) * arr[i];
        }
        else {
            dp[i] = (i+1) * arr[i];
        }
        closestNum.push(i);
        sum = (sum + dp[i]) % mod;
    }
    return sum;
}